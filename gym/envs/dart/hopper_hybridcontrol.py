import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *

import copy

import joblib, os
from pydart2.utils.transformations import quaternion_from_matrix, euler_from_matrix, euler_from_quaternion
from gym import error, spaces


class DartHopperHybridEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        self.action_scale = np.array([200.0, 200.0, 200.0])
        self.train_UP = False
        self.velrew_input = False
        self.noisy_input = False
        self.randomize_initial_state = False
        self.input_time = False

        self.hybrid_control = True
        self.combine_mode = 1  # 0: replace with jt control when in contact, 1: average with jt control when in contact

        self.use_visualization = False

        self.vector_step_sim = True

        self.randomize_history_input = False
        self.history_buffers = []

        self.terminate_for_not_moving \
            = None # [1.0, 1.5]  # [distance, time], need to mvoe distance in time

        obs_dim = 11

        if self.hybrid_control:
            self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])
            self.action_scale = np.array([200.0, 200.0, 200.0])
            obs_dim += 2
            self.current_contact_info = [False, np.zeros(3), None]

        self.actuator_nonlin_coef = 0.0

        self.reward_clipping = 125
        self.resample_task_on_reset = False

        self.velrew_weight = 1.0
        self.angvel_rew = 0.0
        self.angvel_clip = 10.0
        self.alive_bonus = 4.0
        self.energy_penalty = 1e-3
        self.action_bound_penalty = 1.0

        self.height_reward = 0.0

        self.UP_noise_level = 0.0
        self.resample_MP = False  # whether to resample the model paraeters

        self.param_manager = hopperContactMassManager(self)

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)

        self.t = 0

        self.action_buffer = []

        self.total_dist = []

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history

        if self.velrew_input:
            obs_dim += 1

        if self.input_time:
            obs_dim += 1

        frame_skip = 4
        dart_env.DartEnv.__init__(self, ['hopper_capsule.skel', 'hopper_box.skel', 'hopper_ellipsoid.skel'], frame_skip,
                                  obs_dim, self.control_bounds, disableViewer=True)

        self.initial_local_coms = [np.copy(bn.local_com()) for bn in self.robot_skeleton.bodynodes]
        self.initial_coms = [np.copy(bn.com()) for bn in self.robot_skeleton.bodynodes]
        self.default_mass = [bn.mass() for bn in self.robot_skeleton.bodynodes]
        self.default_inertia = [bn.I for bn in self.robot_skeleton.bodynodes]

        self.obstacle_bodynodes = [self.dart_world.skeletons[0].bodynode(bn) for bn in sorted(
            [b.name for b in self.dart_world.skeletons[0].bodynodes if 'obstacle' in b.name])]

        self.visual_bodynodes = [self.dart_world.skeletons[0].bodynode(bn) for bn in sorted(
            [b.name for b in self.dart_world.skeletons[0].bodynodes if 'visual_obj' in b.name])]

        self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_worlds[0].set_collision_detector(3)
        self.dart_worlds[1].set_collision_detector(2)
        self.dart_worlds[2].set_collision_detector(1)

        self.dart_world = self.dart_worlds[0]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 0
        self.act_delay = 0

        self.cycle_times = []  # gait cycle times
        self.previous_contact = None

        # MMHACK: PCA action space
        print("current param ", self.current_param)
        self.param_manager.set_simulator_parameters(self.current_param)

        self.height_threshold_low = 0.56 * self.robot_skeleton.bodynodes[2].com()[1]
        self.rot_threshold = 0.8

        self.short_perturb_params = [] #[1.0, 1.3, np.array([-80, 0, 0])] # start time, end time, force

        print('sim parameters: ', self.param_manager.get_simulator_parameters())
        self.current_param = self.param_manager.get_simulator_parameters()
        self.active_param = self.param_manager.activated_param

        # data structure for actuation modeling
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]

        self.cur_step = 0

        self.add_perturbation = False
        self.perturbation_parameters = [0.01, 25, 2, 50]  # probability, magnitude, bodyid, duration
        self.perturbation_duration = 40
        self.perturb_force = np.array([0, 0, 0])

        utils.EzPickle.__init__(self)

    def jt_control(self, desired_contact_force, contact_offset):
        J = self.robot_skeleton.bodynodes[5].linear_jacobian(contact_offset)
        return np.dot(J.T, desired_contact_force)[3:]

    def resample_task(self, iter_num=None):
        world_selection = 0  # np.random.randint(len(self.dart_worlds))
        self.dart_world = self.dart_worlds[world_selection]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        self.resample_MP = False

        # self.param_manager.resample_parameters()

        self.current_param = self.param_manager.get_simulator_parameters()

        self.action_scale = np.array([200.0, 200, 200]) * np.random.choice([-1, 1])

        return np.array(self.current_param), self.velrew_weight, world_selection, self.action_scale

    def set_task(self, task_params):
        self.dart_world = self.dart_worlds[task_params[2]]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        self.param_manager.set_simulator_parameters(task_params[0])
        self.current_param = self.param_manager.get_simulator_parameters()
        self.velrew_weight = task_params[1]

        self.action_scale = task_params[3]

    def pad_action(self, a):
        full_ac = np.zeros(len(self.robot_skeleton.q))
        full_ac[3:] = a
        return full_ac

    def unpad_action(self, a):
        return a[3:]

    def do_simulation(self, tau, n_frames):
        average_torque = []

        if self.add_perturbation:
            if self.perturbation_duration == 0:
                self.perturb_force *= 0
                if np.random.random() < self.perturbation_parameters[0]:
                    axis_rand = np.random.randint(0, 2, 1)[0]
                    direction_rand = np.random.randint(0, 2, 1)[0] * 2 - 1
                    self.perturb_force[axis_rand] = direction_rand * self.perturbation_parameters[1]
                    self.perturbation_duration = self.perturbation_parameters[3]

            else:
                self.perturbation_duration -= 1

        if not self.vector_step_sim:
            for _ in range(n_frames):
                if self.add_perturbation:
                    self.robot_skeleton.bodynodes[self.perturbation_parameters[2]].add_ext_force(self.perturb_force)


                if len(self.short_perturb_params) > 0:
                    if self.cur_step * self.dt > self.short_perturb_params[0] and \
                            self.cur_step * self.dt < self.short_perturb_params[1]:
                        self.robot_skeleton.bodynodes[2].add_ext_force(self.short_perturb_params[2])

                self.robot_skeleton.set_forces(tau)
                self.joint_works += np.abs(tau * np.array(self.robot_skeleton.dq) * self.sim_dt)
                self.dart_world.step()
        else:
            self.dart_world.step_control(2, tau, n_frames)

    def advance(self, a):
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay - 1]

        self.posbefore = self.robot_skeleton.q[0]
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        if self.hybrid_control:
            contact_forces = clamped_control[3:] * 200.0
            if len(contact_forces) == 2:
                contact_forces = np.concatenate([contact_forces, [0]])
            clamped_control = clamped_control[0:3] * self.action_scale

            if self.current_contact_info[0]:
                contact_centric_control = self.jt_control(contact_forces, self.current_contact_info[1])
                if self.combine_mode == 0:
                    clamped_control = np.clip(contact_centric_control, -1, 1)
                else:
                    clamped_control = np.clip(0.5 * (contact_centric_control + clamped_control), -1, 1)
            tau = np.zeros(self.robot_skeleton.ndofs)
            tau[3:] = clamped_control[0:len(tau[3:])]
        else:
            tau = np.zeros(self.robot_skeleton.ndofs)
            tau[3:] = (clamped_control * self.action_scale)[0:len(tau[3:])]

        self.do_simulation(tau, self.frame_skip)

    def about_to_contact(self):
        return False

    def post_advance(self):
        self.dart_world.check_collision()

    def check_fall_on_ground(self):
        fog = False
        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        permitted_contact_bodies = [self.dart_world.skeletons[0].bodynodes[1],
                                    self.dart_world.skeletons[0].bodynodes[2], self.robot_skeleton.bodynodes[-1],
                                    self.robot_skeleton.bodynodes[-2]]
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
            if contact.bodynode1 not in permitted_contact_bodies and contact.bodynode2 not in permitted_contact_bodies:
                fog = True
        return fog

    def terminated(self):

        self.fall_on_ground = self.check_fall_on_ground()

        s = self.state_vector()
        height = self.robot_skeleton.bodynodes[2].com()[1]
        ang = self.robot_skeleton.q[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (
                    np.abs(self.robot_skeleton.dq) < 100).all() \
                    and not self.fall_on_ground \
                    and (height > self.height_threshold_low) and (abs(ang) < self.rot_threshold))

        if self.terminate_for_not_moving is not None:
            if self.t > self.terminate_for_not_moving[1] and \
                    (np.abs(self.robot_skeleton.q[0]) < self.terminate_for_not_moving[0] or
                     self.robot_skeleton.q[0] * self.velrew_weight < 0):
                done = True

        return done


    def pre_advance(self):
        self.posbefore = self.robot_skeleton.q[0]
        self.q_before = np.array(self.robot_skeleton.q)

    def reward_func(self, a, step_skip=1):
        posafter = self.robot_skeleton.q[0]

        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight
        # reward = np.clip(reward, -100.0, 1.0)

        reward -= np.clip(self.robot_skeleton.dq[2], -self.angvel_clip, self.angvel_clip) * self.angvel_rew

        reward += self.alive_bonus * step_skip

        reward -= self.energy_penalty * np.square(a).sum()

        reward -= 5e-1 * joint_limit_penalty
        reward += self.height_reward * (self.robot_skeleton.q[1] - 0.7)

        reward = np.clip(reward, -np.inf, self.reward_clipping)

        return reward

    def step(self, a):
        # a = -np.copy(a)# + 0.15

        prev_pose = np.array(self.robot_skeleton.q)
        action_bound_violation = np.sum(np.clip(np.abs(a) - 1, 0, 10000))

        self.action_filter_cache.append(a)

        self.t += self.dt
        self.pre_advance()
        self.advance(a)
        reward = self.reward_func(a)
        reward -= self.action_bound_penalty * action_bound_violation

        done = self.terminated()# or self.cur_step >= 999

        if done:
            reward = 0

        ob = self._get_obs()

        self.cur_step += 1

        envinfo = {}

        contacts = self.dart_world.collision_result.contacts
        if self.hybrid_control:
            self.current_contact_info = [False, np.zeros(3), np.zeros(3)]
        for contact in contacts:
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
            if self.hybrid_control:
                if contact.bodynode1 == self.robot_skeleton.bodynodes[5] or contact.bodynode2 == self.robot_skeleton.bodynodes[5]:
                  self.current_contact_info = [True, self.robot_skeleton.bodynodes[5].to_local(contact.point), self.robot_skeleton.bodynodes[2].to_local(contact.point)]
            if self.robot_skeleton.bodynode('h_foot') in [contact.bodynode1, contact.bodynode2] \
                    and \
                    self.dart_world.skeletons[0].bodynodes[0] in [contact.bodynode1, contact.bodynode2]:
                if self.previous_contact is None:
                    self.previous_contact = self.cur_step * self.dt
                else:
                    self.cycle_times.append(self.cur_step * self.dt - self.previous_contact)
                    self.previous_contact = self.cur_step * self.dt
        self.gait_freq = 0

        horizontal_range = [-0.03, 0.03]
        vertical_range = [-0.01, 0.1]
        vertical_range = [-5, -5]
        self.obstacle_bodynodes[0].shapenodes[0].set_offset([0, -100, 0])
        self.obstacle_bodynodes[0].shapenodes[1].set_offset([0, -100, 0])
        for obid in range(1, len(self.obstacle_bodynodes)):
            if self.robot_skeleton.C[0] - 0.6 > self.obstacle_bodynodes[obid].shapenodes[0].offset()[0]:
                last_ob_id = (obid-1 + len(self.obstacle_bodynodes)-2) % (len(self.obstacle_bodynodes)-1)+1
                last_ob_pos = self.obstacle_bodynodes[last_ob_id].shapenodes[0].offset()[0]
                offset = np.copy(self.obstacle_bodynodes[obid].shapenodes[0].offset())

                sampled_v = np.random.uniform(vertical_range[0], vertical_range[1])
                sampled_h = np.random.uniform(horizontal_range[0], horizontal_range[1])

                offset[0] = last_ob_pos + 0.2 + sampled_h
                offset[1] = sampled_v

                self.obstacle_bodynodes[obid].shapenodes[0].set_offset(offset)
                self.obstacle_bodynodes[obid].shapenodes[1].set_offset(offset)

        envinfo['gait_frequency'] = self.gait_freq
        envinfo['xdistance'] = self.robot_skeleton.q[0]
        envinfo['pose'] = prev_pose

        return ob, reward, done, envinfo

    def _get_obs(self, update_buffer=True):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        # state[-1] = self.robot_skeleton.dC[0]
        # state = np.array(self.robot_skeleton.q[1:])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        state[1] = (state[1] + np.pi) % (2 * np.pi) - np.pi

        if self.hybrid_control:
            if self.current_contact_info[0]:
                state = np.concatenate([state, self.current_contact_info[2][0:2]])
            else:
                state = np.concatenate([state, np.zeros(2)])

        if self.noisy_input:
            state = state + np.random.normal(0, .05, len(state))

        if self.train_UP:
            UP = self.param_manager.get_simulator_parameters()
            if self.UP_noise_level > 0:
                UP += np.random.uniform(-self.UP_noise_level, self.UP_noise_level, len(UP))
                UP = np.clip(UP, -0.05, 1.05)
            state = np.concatenate([state, UP])

        if self.input_time:
            state = np.concatenate([state, [self.t]])

        if update_buffer:
            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        current_obs = np.array([]) # the current part of observation
        history_obs = np.array([]) # the history part of the observation
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
                if i == 0:
                    current_obs = np.concatenate([history_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
                if i > 0 and self.randomize_history_input:
                    history_obs = np.concatenate([history_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0] * 0.0])
                if i == 0:
                    current_obs = np.concatenate([history_obs, self.observation_buffer[0] * 0.0])
                if i > 0 and self.randomize_history_input:
                    history_obs = np.concatenate([history_obs, self.observation_buffer[0] * 0.0])
        if self.randomize_history_input:
            self.history_buffers.append(history_obs)
            final_obs = np.concatenate([current_obs, self.history_buffers[np.random.randint(len(self.history_buffers))]])


        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[-1 - i]])
            else:
                final_obs = np.concatenate([final_obs, [0.0] * len(self.control_bounds[0])])

        if self.velrew_input:
            final_obs = np.concatenate([final_obs, [self.velrew_weight]])

        return final_obs

    def get_lowdim_obs(self):
        full_obs = self._get_obs(update_buffer=False)
        return np.array([full_obs[1], full_obs[2]])

    def reset_model(self):
        if self.resample_task_on_reset:
            self.resample_task()
        for world in self.dart_worlds:
            world.reset()
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005,
                                                              size=self.robot_skeleton.ndofs) if self.randomize_initial_state else np.zeros(
            self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005,
                                                               size=self.robot_skeleton.ndofs) if self.randomize_initial_state else np.zeros(
            self.robot_skeleton.ndofs)


        self.set_state(qpos, qvel)

        if self.resample_MP:
            self.param_manager.resample_parameters()
        self.current_param = self.param_manager.get_simulator_parameters()

        self.observation_buffer = []
        self.action_buffer = []

        self.action_filter_cache = []

        state = self._get_obs(update_buffer=True)

        # setup obstacle
        horizontal_range = [0.5, 0.5]
        vertical_range = [-0.01, 0.1]
        vertical_range = [-5, -5]
        for obid in range(1, len(self.obstacle_bodynodes)):
            sampled_v = np.random.uniform(vertical_range[0], vertical_range[1])
            sampled_h = np.random.uniform(horizontal_range[0], horizontal_range[1]) + 0.0 + 0.2 * obid
            self.obstacle_bodynodes[obid].shapenodes[0].set_offset([sampled_h, sampled_v, 0])
            self.obstacle_bodynodes[obid].shapenodes[1].set_offset([sampled_h, sampled_v, 0])
            self.dart_world.skeletons[0].dofs[obid].set_damping_coefficient(1000.0)
            self.dart_world.skeletons[0].dofs[obid].set_spring_stiffness(500000.0)

        self.action_buffer = []

        self.cur_step = 0

        self.t = 0

        self.fall_on_ground = False

        self.cycle_times = []  # gait cycle times
        self.previous_contact = None

        self.accumulated_reward = 0

        self.joint_works = np.zeros(len(self.robot_skeleton.dq))

        if self.hybrid_control:
            self.current_contact_info = [False, np.zeros(3), np.zeros(3)]

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4
        self.render_frame = [0, 500, 400, 900]

    def state_vector(self):
        s = np.copy(np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]))
        s[1] += self.zeroed_height

        return s

    def set_state_vector(self, s):
        snew = np.copy(s)
        snew[1] -= self.zeroed_height
        self.robot_skeleton.q = snew[0:len(self.robot_skeleton.q)]
        self.robot_skeleton.dq = snew[len(self.robot_skeleton.q):]

    def set_sim_parameters(self, pm):
        self.param_manager.set_simulator_parameters(pm)

    def get_sim_parameters(self):
        return self.param_manager.get_simulator_parameters()

    def resample_reward_function(self):
        # alive bonus, velocity reward, energy penalty
        coef = np.random.random(3)
        self.alive_bonus = coef[0] * 5.0  # 0-5
        self.energy_penalty = coef[1] * 0.01  # 0-0.01
        self.velrew_weight = coef[2] * 5.0  # 0-5
        return coef

    def set_reward_function(self, coef):
        self.alive_bonus = coef[0] * 5.0  # 0-5
        self.energy_penalty = coef[1] * 0.01  # 0-0.01
        self.velrew_weight = coef[2] * 5.0  # 0-5

    # set state from observation
    def set_from_obs(self, obs):
        xpos = self.robot_skeleton.q[0] + obs[5] * 0.008
        state = np.concatenate([[xpos], obs[:-1]])
        self.set_state_vector(state)
