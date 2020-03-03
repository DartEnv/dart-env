__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.parameter_managers import CartPoleManager
import os
import joblib
from gym.envs.dart.sub_tasks import *
import copy


class DartCartPoleSwingUpEnv(dart_env.DartEnv):
    def __init__(self):
        self.control_bounds = np.array([[1.0], [-1.0]])
        self.action_scale = np.array([40])
        self.train_UP = False
        self.resample_MP = False  # whether to resample the model parameters
        self.train_mp_sel = False
        self.perturb_MP = False
        self.avg_div = 0
        self.param_manager = CartPoleManager(self)

        self.use_discrete_action = False

        self.input_exp = 0 # dim of input exploration

        self.use_disc_ref_policy = False
        self.disc_ref_weight = 0.001
        self.disc_funcs = [] # vfunc, obs_disc, act_disc, state_filter, state_unfilter
        self.disc_policy = None
        self.disc_fit_policy = None
        self.learn_additive_pol = False

        self.randomize_initial_state = True

        self.split_task_test = False
        self.tasks = TaskList(1)
        self.tasks.add_world_choice_tasks([0])
        # self.tasks.add_range_param_tasks([0, [[0.0, 0.5], [0.5, 1.0], [0.0, 0.5], [0.5, 1.0]]])
        # self.tasks.add_range_param_tasks([2, [[0.0, 0.5], [0.5, 1.0], [0.5, 1.0], [0.0, 0.5]]])

        self.cur_step = 0

        obs_dim = 4 + self.input_exp
        if self.train_UP:
            obs_dim += self.param_manager.param_dim
        if self.train_mp_sel:
            obs_dim += 5

        self.juggling = False
        if self.juggling:
            obs_dim += 4
            self.action_scale *= 2

        if self.learn_additive_pol:
            obs_dim += 1

        self.dyn_models = [None]
        self.dyn_model_id = 0
        self.base_path = None
        self.transition_locator = None

        if self.split_task_test:
            obs_dim += self.tasks.task_input_dim()
        if self.avg_div > 1:
            obs_dim += self.avg_div

        if self.use_discrete_action:
            from gym import spaces
            self.action_space = spaces.Discrete(5)

        dart_env.DartEnv.__init__(self, ['cartpole_swingup.skel', 'cartpole_swingup_variation1.skel',
                                         'cartpole_swingup_variation2.skel'], 2, obs_dim, self.control_bounds, dt=0.01,
                                  disableViewer=True, action_type="continuous" if not self.use_discrete_action else "discrete")
        self.current_param = self.param_manager.get_simulator_parameters()
        # self.dart_world.skeletons[1].bodynodes[0].set_friction_coeff(0.2)
        # self.dart_world.skeletons[1].bodynodes[0].set_restitution_coeff(0.7)
        # self.dart_world.skeletons[-1].bodynodes[-1].set_restitution_coeff(0.7)

        self.dart_world = self.dart_worlds[0]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        # info for building gnn for dynamics
        self.ignore_joint_list = [2]
        self.ignore_body_list = [2]
        self.joint_property = []  # what to include in the joint property part
        self.bodynode_property = ['mass']
        self.root_type = 'None'
        self.root_id = 0

        self.use_qdqstate = True

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 1
        self.act_delay = 1

        utils.EzPickle.__init__(self)

    def resample_task(self):
        self.param_manager.resample_parameters()
        self.current_param = self.param_manager.get_simulator_parameters()
        return self.current_param

    def set_task(self, task_params):
        self.param_manager.set_simulator_parameters(task_params)

    def about_to_contact(self):
        return False

    def pad_action(self, a):
        full_ac = np.zeros(len(self.robot_skeleton.q))
        full_ac[0] = a[0]
        return full_ac

    def unpad_action(self, a):
        return [a[0]]

    def terminated(self):
        done = abs(self.robot_skeleton.dq[1]) > 35 or abs(self.robot_skeleton.q[0]) > 2.0
        return done

    def reward_func(self, a):
        ang = self.robot_skeleton.q[1]

        ang_proc = (np.abs(ang) % (2 * np.pi))
        ang_proc = np.min([ang_proc, (2 * np.pi) - ang_proc])

        if not self.juggling:
            alive_bonus = 6.0
        else:
            alive_bonus = 4.0
        ang_cost = 1.0 * (ang_proc ** 2)
        quad_ctrl_cost = 0.01 * np.square(a).sum()
        com_cost = 2.0 * np.abs(self.robot_skeleton.q[0]) ** 2

        reward = alive_bonus - ang_cost - quad_ctrl_cost - com_cost
        if ang_proc < 0.5:
            reward += np.max([5 - (ang_proc) * 4, 0]) + np.max([3 - np.abs(self.robot_skeleton.dq[1]), 0])
        return reward

    def step(self, a):
        if self.use_discrete_action:
            a = np.array([a * 1.0/ np.floor(self.action_space.n/2.0) - 1.0])

        self.advance(a)

        ob = self._get_obs()

        reward = self.reward_func(a)

        done = self.terminated()

        if self.perturb_MP:
            self.param_manager.set_simulator_parameters(
                self.current_param + np.random.uniform(-0.01, 0.01, len(self.current_param)))

        self.cur_step += 1

        envinfo = {'model_parameters': self.param_manager.get_simulator_parameters()
            , 'dyn_model_id': self.dyn_model_id, 'state_index': self.state_index}

        return ob, reward, done, envinfo

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
        tau[0] = clamped_control[0] * self.action_scale
        self.do_simulation(tau, self.frame_skip)


    def _get_obs(self):
        state = np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

        ang = self.robot_skeleton.q[1]
        ang_proc = (ang % (2 * np.pi))
        if ang_proc > np.pi:
            ang_proc -= 2 * np.pi

        state[1] = ang_proc

        if self.juggling:
            state = np.concatenate([state, self.dart_world.skeletons[1].com()[[0, 1]],
                                    self.dart_world.skeletons[1].com_velocity()[[0, 1]]])  # ,\
            # self.dart_world.skeletons[2].com()[[0, 1]], self.dart_world.skeletons[2].com_velocity()[[0, 1]]])

        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])

        if self.train_mp_sel:
            state = np.concatenate([state, [self.rand]])

        if self.split_task_test:
            state = np.concatenate([state, self.tasks.get_task_inputs(self.state_index)])

        if self.learn_additive_pol:
            # take action predicted by discrete policy and append to the observation
            if self.disc_fit_policy is not None:
                st = self.disc_funcs[3](self.state_vector())
                disc_act = self.disc_fit_policy.pred(st)[0]
                state = np.concatenate([state, disc_act])
            else:
                if self.disc_funcs[1](self.disc_funcs[3](self.state_vector())) in self.disc_policy:
                    sid = self.disc_funcs[1](self.disc_funcs[3](self.state_vector()))
                    disc_act = self.disc_funcs[2].get_midstate(self.disc_policy[sid])
                    state = np.concatenate([state, disc_act])
                else:
                    state = np.concatenate([state, [0]])

        if self.input_exp > 0:
            state = np.concatenate([state, self.sampled_input_exp])

        self.observation_buffer.append(np.copy(state))
        if len(self.observation_buffer) < self.obs_delay + 1:
            state = self.observation_buffer[0]
        else:
            state = self.observation_buffer[-self.obs_delay - 1]

        return state

    def reset_model(self):
        self.total_dist = []
        self.dart_world.reset()
        qpos = self.robot_skeleton.q
        qvel = self.robot_skeleton.dq
        if self.randomize_initial_state:
            qpos += self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
            qvel += self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
            if self.np_random.uniform(0, 1) > 0.5:
                qpos[1] += np.pi
            else:
                qpos[1] += -np.pi
        else:
            qpos[1] += np.pi

        # qpos[1]+=self.np_random.uniform(low=-np.pi, high=np.pi, size=1)

        if self.resample_MP:
            self.param_manager.resample_parameters()

        self.current_param = self.param_manager.get_simulator_parameters()
        self.state_index = self.dyn_model_id

        if self.input_exp > 0:
            self.sampled_input_exp = np.random.random(self.input_exp) # uniform distribution

        if self.train_UP:
            self.state_index = 0
            if len(self.param_manager.get_simulator_parameters()) > 1:
                if self.param_manager.get_simulator_parameters()[0] < 0.5 and \
                                self.param_manager.get_simulator_parameters()[1] >= 0.5:
                    self.state_index = 1
                elif self.param_manager.get_simulator_parameters()[0] >= 0.5 and \
                                self.param_manager.get_simulator_parameters()[1] < 0.5:
                    self.state_index = 2
                elif self.param_manager.get_simulator_parameters()[0] >= 0.5 and \
                                self.param_manager.get_simulator_parameters()[1] >= 0.5:
                    self.state_index = 3
            if self.upselector is not None:
                self.state_index = self.upselector.classify([self.param_manager.get_simulator_parameters()],
                                                            stoch=False)

        if not self.juggling:
            self.dart_world.skeletons[1].set_positions([0, 0, 0, 100, 0, 0])
            self.dart_world.skeletons[2].set_positions([0, 0, 0, 200, 0, 0])
        else:
            self.jug_pos = self.dart_world.skeletons[1].q + self.np_random.uniform(low=-.1, high=.1, size=6)
            self.jug_vel = self.dart_world.skeletons[1].dq + self.np_random.uniform(low=-.01, high=.01, size=6)
            self.jug_pos[-1] = 0
            self.jug_vel[-1] = 0
            self.jug_pos[-3] = self.np_random.uniform(low=0.05, high=0.5)
            self.dart_world.skeletons[1].set_positions(self.jug_pos)
            self.dart_world.skeletons[1].set_velocities(self.jug_vel)
            # self.dart_world.skeletons[2].set_positions([0,0,0,200, 0, 0])
            self.jug_pos2 = self.dart_world.skeletons[2].q + self.np_random.uniform(low=-.1, high=.1, size=6)
            self.jug_vel2 = self.dart_world.skeletons[2].dq + self.np_random.uniform(low=-.01, high=.01, size=6)
            self.jug_pos2[-1] = 0
            self.jug_vel2[-1] = 0
            self.jug_pos2[-3] = self.np_random.uniform(low=-0.5, high=-0.05)
            self.jug_pos2[-3] = 100
            self.dart_world.skeletons[2].set_positions(self.jug_pos2)
            self.dart_world.skeletons[2].set_velocities(self.jug_vel2)

        if self.split_task_test:
            self.state_index = np.random.randint(self.tasks.task_num)
            world_choice, pm_id, pm_val, jt_id, jt_val = self.tasks.resample_task(self.state_index)
            if self.dart_world != self.dart_worlds[world_choice]:
                self.dart_world = self.dart_worlds[world_choice]
                self.robot_skeleton = self.dart_world.skeletons[-1]
                self.dart_world.skeletons[1].set_positions([0, 0, 0, 100, 0, 0])
                self.dart_world.skeletons[2].set_positions([0, 0, 0, 200, 0, 0])
                qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
                qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01,
                                                                       size=self.robot_skeleton.ndofs)
                if self.np_random.uniform(0, 1) > 0.5:
                    qpos[1] += np.pi
                else:
                    qpos[1] += -np.pi
                self.set_state(qpos, qvel)
                if not self.disableViewer:
                    self._get_viewer().sim = self.dart_world
            self.param_manager.controllable_param = pm_id
            self.param_manager.set_simulator_parameters(np.array(pm_val))
            for ind, jtid in enumerate(jt_id):
                self.robot_skeleton.joints[jtid].set_position_upper_limit(0, jt_val[ind][1])
                self.robot_skeleton.joints[jtid].set_position_lower_limit(0, jt_val[ind][0])

        if self.train_mp_sel:
            self.rand = np.random.random()
        self.set_state(qpos, qvel)

        # if self.base_path is not None and self.dyn_model_id != 0:
        #    base_len = len(self.base_path)
        #    base_state = self.base_path['env_infos']['state_act'][np.random.randint(base_len-3)][0:len(self.state_vector())]
        #    self.set_state_vector(base_state + self.np_random.uniform(low=-0.01, high=0.01, size=len(base_state)))

        self.cur_step = 0

        self.observation_buffer = []
        self.action_buffer = []

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0

    def state_vector(self):
        return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq])

    def set_state_vector(self, s):
        self.robot_skeleton.q = s[0:len(self.robot_skeleton.q)]
        self.robot_skeleton.dq = s[len(self.robot_skeleton.q):]

    def set_sim_parameters(self, pm):
        self.param_manager.set_simulator_parameters(pm)

    def get_sim_parameters(self):
        return self.param_manager.get_simulator_parameters()

    def pre_advance(self):
        pass

    def post_advance(self):
        pass
