import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.parameter_managers import *
from gym.envs.dart.sub_tasks import *
import copy


class DartWalker2dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*6,[-1.0]*6])
        #self.control_bounds[1][1] = -0.3
        #self.control_bounds[1][4] = -0.3
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])
        obs_dim = 17
        self.train_UP = False
        self.noisy_input = True
        self.resample_MP = False
        self.UP_noise_level = 0.0
        self.input_time = False
        self.param_manager = walker2dParamManager(self)

        self.diff_obs = False

        self.terminate_for_not_moving = None#[1.0, 1.5]  # [distance, time], need to mvoe distance in time

        self.vibrating_ground = False
        self.ground_vib_params = [0.12,1.5] # magnitude, frequency
        self.randomize_ground_vib = False
        self.ground_vib_input = False

        self.action_filtering = 0  # window size of filtering, 0 means no filtering
        self.action_filter_cache = []
        self.action_filter_in_env = False  # whether to filter out actions in the environment
        self.action_filter_inobs = False  # whether to add the previous actions to the observations
        self.action_filter_reward = 0.0
        self.action_filter_delta_act = 0.0 # 0.0 if disabled

        self.velrew_weight = 1.0

        if self.action_filtering > 0 and self.action_filter_inobs:
            obs_dim += len(self.action_scale) * self.action_filtering

        self.avg_div = 0
        self.target_vel = 0.9
        self.split_task_test = False
        self.tasks = TaskList(2)
        self.tasks.add_world_choice_tasks([0, 0])
        self.learn_forwardbackward = False
        self.task_expand_flag = False
        self.state_index = 0
        self.use_sparse_reward = False

        self.t = 0

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history


        obs_perm_base = np.array(
            [0.0001, 1, 5, 6, 7, 2, 3, 4, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        act_perm_base = np.array([3, 4, 5, 0.0001, 1, 2])
        self.obs_perm = np.copy(obs_perm_base)

        for i in range(self.include_obs_history - 1):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(obs_perm_base) * (np.abs(obs_perm_base) + len(self.obs_perm))])
        for i in range(self.include_act_history):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(act_perm_base) * (np.abs(act_perm_base) + len(self.obs_perm))])
        self.act_perm = np.array([3, 4, 5, 0.0001, 1, 2])

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)
            self.obs_perm = np.concatenate([self.obs_perm, np.arange(int(len(self.obs_perm)),
                                                int(len(self.obs_perm)+len(self.param_manager.activated_param)))])

        if self.ground_vib_input:
            obs_dim += 2
            self.obs_perm = np.concatenate([self.obs_perm, np.arange(int(len(self.obs_perm)),
                                                                     int(len(self.obs_perm) + 2))])

        if self.diff_obs:
            obs_dim = obs_dim * obs_dim

        if self.input_time:
            obs_dim += 1

        dart_env.DartEnv.__init__(self, ['walker2d.skel', 'walker2d_variation1.skel'\
                                         , 'walker2d_variation2.skel'], 4, obs_dim, self.control_bounds, disableViewer=True)

        # data structure for actuation modeling
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]

        self.dart_worlds[0].set_collision_detector(3)
        self.dart_worlds[1].set_collision_detector(0)
        self.dart_worlds[2].set_collision_detector(1)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]
        if not self.disableViewer:
            self._get_viewer().sim = self.dart_world

        # info for building gnn for dynamics
        self.ignore_joint_list = []
        self.ignore_body_list = [0, 1]
        self.joint_property = ['limit']  # what to include in the joint property part
        self.bodynode_property = ['mass']
        self.root_type = 'None'
        self.root_id = 0

        self.cur_step = 0

        # no joint limit
        '''for world in self.dart_worlds:
            for skeleton in world.skeletons:
                for jt in range(0, len(skeleton.joints)):
                    for dof in range(len(skeleton.joints[jt].dofs)):
                        if skeleton.joints[jt].has_position_limit(dof):
                            skeleton.joints[jt].set_position_limit_enforced(False)'''

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.state_buffer = []

        self.cycle_times = [] # gait cycle times
        self.previous_contact = None # [left or right foot, time]

        self.obs_delay = 0
        self.act_delay = 0

        #print('sim parameters: ', self.param_manager.get_simulator_parameters())
        self.current_param = self.param_manager.get_simulator_parameters()

        utils.EzPickle.__init__(self)

    def resample_task(self):
        self.param_manager.resample_parameters()
        self.current_param = self.param_manager.get_simulator_parameters()
        #self.velrew_weight = np.sign(np.random.randn(1))[0]
        return np.array(self.current_param), self.velrew_weight

    def set_task(self, task_params):
        self.param_manager.set_simulator_parameters(task_params[0])
        self.velrew_weight = task_params[1]

    def about_to_contact(self):
        return False

    def pad_action(self, a):
        full_ac = np.zeros(len(self.robot_skeleton.q))
        full_ac[3:] = a
        return full_ac

    def pre_advance(self):
        self.posbefore = self.robot_skeleton.q[0]

    def terminated(self):
        s = self.state_vector()
        height = self.robot_skeleton.bodynodes[2].com()[1]
        ang = self.robot_skeleton.q[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                   (height > .8) and (height < 2.0) and (abs(ang) < 1.0))
        lfooth = self.robot_skeleton.bodynode('h_foot_left').C[1]
        rfooth = self.robot_skeleton.bodynode('h_foot').C[1]
        #if lfooth > self.robot_skeleton.bodynode('h_thigh').C[1] or rfooth > self.robot_skeleton.bodynode('h_thigh_left').C[1]: # don't allow the feet to raise high
        #    done = True
        if self.cur_step >= 1000:
            done = True

        if self.terminate_for_not_moving is not None:
            if self.t > self.terminate_for_not_moving[1] and \
                    (np.abs(self.robot_skeleton.q[0]) < self.terminate_for_not_moving[0] or
                     self.robot_skeleton.q[0] * self.velrew_weight < 0):
                done = True

        return done

    def post_advance(self):
        pass

    def reward_func(self, a, step_skip=1, sparse=False):
        posafter, ang = self.robot_skeleton.q[0, 2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        alive_bonus = 1.0 * step_skip
        vel = (posafter - self.posbefore) / self.dt * self.velrew_weight
        reward = vel
        reward += alive_bonus
        reward -= 1e-1 * np.square(a).sum()
        joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        reward -= 5e-1 * joint_limit_penalty

        if sparse:
            reward = 0.0
            if self.terminated():
                reward = self.robot_skeleton.q[0]

        return reward

    def advance(self, a):
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay - 1]

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale
        self.posbefore = self.robot_skeleton.q[0]

        # compensate for gravity
        #tau[1] = self.robot_skeleton.mass() * 9.81

        self.do_simulation(tau, self.frame_skip)

    def step(self, a):
        action_filter_rew = 0
        if self.action_filtering > 0 and self.action_filter_reward > 0 and len(self.action_filter_cache) > 0:
            action_filter_rew = -np.abs(np.mean(self.action_filter_cache, axis=0) - a).sum() * self.action_filter_reward

        if self.vibrating_ground:
            self.dart_world.skeletons[0].joints[0].set_rest_position(0, self.ground_vib_params[0] * np.sin(2*np.pi*self.ground_vib_params[1] * self.cur_step * self.dt))

        if self.action_filter_delta_act > 0:
            if len(self.action_filter_cache) == 0:
                base_a = np.zeros(len(a))
            else:
                base_a = np.mean(self.action_filter_cache, axis=0)
            act = base_a + a * self.action_filter_delta_act
            self.action_filter_cache.append(act)
            a = np.copy(act)
        else:
            self.action_filter_cache.append(a)
        if len(self.action_filter_cache) > self.action_filtering:
            self.action_filter_cache.pop(0)
        if self.action_filtering > 0 and self.action_filter_in_env:
            a = np.mean(self.action_filter_cache, axis=0)

        contacts = self.dart_world.collision_result.contacts
        for contact in contacts:
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
            if self.robot_skeleton.bodynode('h_foot_left') in [contact.bodynode1, contact.bodynode2]\
                    and \
                    self.dart_world.skeletons[0].bodynodes[0] in [contact.bodynode1, contact.bodynode2]:
                if self.previous_contact is None:
                    self.previous_contact = [0, self.cur_step * self.dt]
                elif self.previous_contact[0] == 1 and self.robot_skeleton.q[3] < self.robot_skeleton.q[6]:
                    self.cycle_times.append(self.cur_step * self.dt - self.previous_contact[1])
                    self.previous_contact = [0, self.cur_step * self.dt]

            if self.robot_skeleton.bodynode('h_foot') in [contact.bodynode1, contact.bodynode2]\
                    and \
                    self.dart_world.skeletons[0].bodynodes[0] in [contact.bodynode1, contact.bodynode2]:
                if self.previous_contact is None:
                    self.previous_contact = [1, self.cur_step * self.dt]
                elif self.previous_contact[0] == 0 and self.robot_skeleton.q[3] > self.robot_skeleton.q[6]:
                    self.cycle_times.append(self.cur_step * self.dt - self.previous_contact[1])
                    self.previous_contact = [1, self.cur_step * self.dt]

        self.cur_step += 1
        self.t += self.dt
        self.advance(a)
        reward = self.reward_func(a, sparse=self.use_sparse_reward) + action_filter_rew

        done = self.terminated()

        ob = self._get_obs()

        self.gait_freq = 0
        # if len(self.cycle_times) > 0:
        #     self.gait_freq = 1.0 / (np.mean(self.cycle_times) * 2)

        return ob, reward, done, {'dyn_model_id':0, 'state_index':self.state_index, 'gait_frequency':self.gait_freq}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.action_filtering > 0 and self.action_filter_inobs:
            state = np.concatenate([state] + self.action_filter_cache)

        self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0] * 0.0])

        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[-1 - i]])
            else:
                final_obs = np.concatenate([final_obs, [0.0] * len(self.control_bounds[0])])

        if self.train_UP:
            UP_parameters = self.param_manager.get_simulator_parameters()
            noise_range = 0.5 / (1.0+np.exp(-20.0*self.UP_noise_level+10.0))
            perturbed_pm = np.copy(UP_parameters)
            if self.UP_noise_level > 0.0:
                for updim in range(len(perturbed_pm)-1): # noise parameter should always be the last one, so no noise added
                    lb = np.clip(perturbed_pm[updim] - noise_range, 0, 1)
                    ub = np.clip(perturbed_pm[updim] + noise_range, 0, 1)
                    perturbed_pm[updim] = np.random.uniform(lb, ub)
            final_obs = np.concatenate([final_obs, perturbed_pm])

        if self.ground_vib_input:
            final_obs = np.concatenate([final_obs, self.ground_vib_params])

        if self.input_time:
            final_obs = np.concatenate([final_obs, [self.t]])

        if self.noisy_input:
            final_obs = final_obs + np.random.normal(0, .01, len(final_obs))

        if self.diff_obs:
            single_obs = np.copy(final_obs)
            for i in range(len(single_obs)-1):
                final_obs = np.concatenate([final_obs, single_obs - np.roll(single_obs, i+1)])


        return final_obs

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.00015, high=.00015, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.00015, high=.00015, size=self.robot_skeleton.ndofs)
        # if np.random.random() > 0.5:
        #     qpos[3] += 0.1
        # else:
        #     qpos[6] += 0.1
        #MMHACK
        qpos[3] += 0.3

        self.set_state(qpos, qvel)

        if self.resample_MP:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()

        if self.split_task_test:
            if self.task_expand_flag:
                self.tasks.expand_range_param_tasks()
                self.task_expand_flag = False
            self.state_index = np.random.randint(self.tasks.task_num)
            world_choice, pm_id, pm_val, jt_id, jt_val = self.tasks.resample_task(self.state_index)
            if self.dart_world != self.dart_worlds[world_choice]:
                self.dart_world = self.dart_worlds[world_choice]
                self.robot_skeleton = self.dart_world.skeletons[-1]
                qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.00015, high=.00015, size=self.robot_skeleton.ndofs)
                qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.00015, high=.00015, size=self.robot_skeleton.ndofs)
                self.set_state(qpos, qvel)
                if not self.disableViewer:
                    self._get_viewer().sim = self.dart_world
            self.param_manager.controllable_param = pm_id
            self.param_manager.set_simulator_parameters(np.array(pm_val))
            for ind, jtid in enumerate(jt_id):
                self.robot_skeleton.joints[jtid].set_position_upper_limit(0, jt_val[ind][1])
                self.robot_skeleton.joints[jtid].set_position_lower_limit(0, jt_val[ind][0])

        self.observation_buffer = []
        self.action_buffer = []
        self.cur_step = 0
        self.t = 0

        self.action_filter_cache = []
        if self.action_filtering > 0:
            for i in range(self.action_filtering):
                self.action_filter_cache.append(np.zeros(len(self.action_scale)))

        self.cycle_times = []  # gait cycle times
        self.previous_contact = None

        if self.randomize_ground_vib:
            frequency = np.random.random() * 4 + 1 # take frequency between 1 - 5
            magnitutde = 0.126 / frequency * (1 + np.random.uniform(-1, 1) / 8.0)
            self.ground_vib_params = [magnitutde, frequency]

        if self.vibrating_ground:
            self.ground_vib_params[0] = np.random.random() * 0.12

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4

    def state_vector(self):
        s = np.copy(np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]))
        s[1] += self.zeroed_height
        return s

    def set_state_vector(self, s):
        snew = np.copy(s)
        snew[1] -= self.zeroed_height
        self.robot_skeleton.q = snew[0:len(self.robot_skeleton.q)]
        self.robot_skeleton.dq = snew[len(self.robot_skeleton.q):]