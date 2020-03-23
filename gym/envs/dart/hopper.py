import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *
from gym.envs.dart.sub_tasks import *
from gym.envs.dart.spline import *
from gym.envs.dart.action_filter import *
import copy

import joblib, os
from pydart2.utils.transformations import quaternion_from_matrix, euler_from_matrix, euler_from_quaternion
from gym import error, spaces


class DartHopperEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        self.action_scale = np.array([200.0, 200.0, 200.0]) * 0.4
        self.train_UP = False
        self.velrew_input = False
        self.noisy_input = False
        self.randomize_initial_state = True
        self.input_time = False
        self.two_pose_input = False  # whether to use two q's instead of q and dq

        self.butterworth_filter = False

        self.randomize_history_input = False
        self.history_buffers = []

        self.randomize_obstacles = False

        self.staged_reward = False
        self.learn_getup = False
        self.learn_alive = False

        self.sparse_reward = False

        self.void_area = False

        self.obs_ranges = np.array([[0.6, 1.8], [-0.5, 0.5], [-2.6, 0.0], [-2.6, 0.0], [-0.8, 0.8],
                                    [-5.0, 5.0], [-5.0, 5.0], [-10.0, 10.0] ,[-20.0, 20.0] ,[-20.0, 20.0], [-20.0, 20.0]])
        self.quantized_observation = False
        self.quantization_param = [3, 50, 1.0]  # min, max, current

        self.pid_controller = None#[500, 5]
        if self.pid_controller is not None:
            self.torque_limit = [-200, 200]
            self.action_scale = np.array([2.0, 2.0, 2.0])

            # MMHACK: PCA action space
            # self.action_scale = np.array([3.0, 3.0])
            # self.control_bounds = np.array([[1.0, 1.0], [-1.0, -1.0]])
            # import pickle
            # fullpath = os.path.join(os.path.dirname(__file__), "models", "pcamodel.pkl")
            # self.pca = pickle.load(open(fullpath, 'rb'))

        self.use_spline_action = False
        if self.use_spline_action:
            self.spline_node_num = 3
            self.spline = CatmullRomSpline(3, self.spline_node_num, False)
            dim = len(self.spline.get_current_parameters())
            self.control_bounds = np.array([[1.0] * dim, [-1.0] * dim])
            self.spline_duration = 0.06

        self.pseudo_lstm_dim = 0  # Number of pseudo lstm hidden size.
        self.diff_obs = False

        self.fallstates = []

        self.terminate_for_not_moving \
            = None#[1.0, 1.5]  # [distance, time], need to mvoe distance in time

        self.action_filtering = 0  # window size of filtering, 0 means no filtering
        self.action_filter_cache = []
        self.action_filter_in_env = False  # whether to filter out actions in the environment
        self.action_filter_inobs = False  # whether to add the previous actions to the observations

        obs_dim = 11

        if self.two_pose_input:
            self.pose_history = []
            obs_dim = 10

        self.obs_projection_model = None

        self.append_zeros = 0

        obs_dim += self.append_zeros

        self.reward_clipping = 125
        self.test_jump_obstacle = False
        self.learn_backflip = False
        self.learn_jump = False
        self.input_obs_height = False
        self.learn_goto = None#np.array([1, 0.3, 0.3])
        self.resample_task_on_reset = False
        self.vibrating_ground = False
        self.ground_vib_params = [0.14, 1.5]  # magnitude, frequency

        self.periodic_noise = False
        self.periodic_noise_params = [0.1, 4.5]  # magnitude, frequency

        self.joint_works = []

        self.learnable_perturbation = False
        self.learnable_perturbation_list = [['h_shin', 80, 0]]  # [bodynode name, force magnitude, torque magnitude
        self.learnable_perturbation_space = spaces.Box(np.array([-1] * len(self.learnable_perturbation_list) * 6),
                                                       np.array([1] * len(self.learnable_perturbation_list) * 6))
        self.learnable_perturbation_act = np.zeros(len(self.learnable_perturbation_list) * 6)

        self.velrew_weight = 1.0
        self.angvel_rew = 0.0
        self.angvel_clip = 10.0
        self.alive_bonus = 1.0
        self.energy_penalty = 1e-3
        self.action_bound_penalty = 1.0

        self.UP_noise_level = 0.0
        self.resample_MP = False  # whether to resample the model paraeters

        self.actuator_nonlinearity = False
        self.actuator_nonlin_coef = 1.0

        self.param_manager = hopperContactMassManager(self)

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)

        if self.action_filtering > 0 and self.action_filter_inobs:
            obs_dim += len(self.action_scale) * self.action_filtering

        if self.test_jump_obstacle:
            obs_dim += 1
            if self.input_obs_height:
                obs_dim += 1
                self.obs_height = 0.0

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

        if self.quantized_observation:
            obs_dim += 3
            self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]])
            mult = 1.0
            self.action_scale = np.array([200.0*mult, 200.0*mult, 200.0*mult, 1.0])
            self.max_quant_budget = 50.0

        self.action_bound_model = None

        if self.pseudo_lstm_dim > 0:
            obs_dim += self.pseudo_lstm_dim * 2
            new_ub = np.concatenate([self.control_bounds[0], np.ones(self.pseudo_lstm_dim * 2)])
            new_lb = np.concatenate([self.control_bounds[1], np.ones(self.pseudo_lstm_dim * 2) * -1])
            self.control_bounds = np.array([new_ub, new_lb])
            self.hidden_states = np.zeros(self.pseudo_lstm_dim * 2)

        if self.diff_obs:
            obs_dim = obs_dim * obs_dim

        frame_skip = 4
        if self.use_spline_action:
            frame_skip = int(self.spline_duration / 0.002)
        dart_env.DartEnv.__init__(self, ['hopper_capsule.skel', 'hopper_box.skel', 'hopper_ellipsoid.skel'], frame_skip,
                                  obs_dim, self.control_bounds, disableViewer=True)

        self.initial_local_coms = [np.copy(bn.local_com()) for bn in self.robot_skeleton.bodynodes]
        self.initial_coms = [np.copy(bn.com()) for bn in self.robot_skeleton.bodynodes]

        self.obstacle_bodynodes = [self.dart_world.skeletons[0].bodynode(bn) for bn in sorted(
            [b.name for b in self.dart_world.skeletons[0].bodynodes if 'obstacle' in b.name])]

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
        self.param_manager.set_simulator_parameters(self.current_param)

        self.height_threshold_low = 0.56 * self.robot_skeleton.bodynodes[2].com()[1]
        self.rot_threshold = 0.4

        if self.staged_reward:
            self.rot_threshold = 1.0
            self.height_threshold_low = 0.36 * self.robot_skeleton.bodynodes[2].com()[1]

        self.short_perturb_params = []  # [1.0, 1.3, np.array([-200, 0, 0])] # start time, end time, force

        print('sim parameters: ', self.param_manager.get_simulator_parameters())
        self.current_param = self.param_manager.get_simulator_parameters()
        self.active_param = self.param_manager.activated_param

        # data structure for actuation modeling
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]

        self.cur_step = 0

        self.stop_velocity_reward = 1000.0
        self.height_penalty = 0.0
        self.obstacle_x_offset = 2.0
        if self.test_jump_obstacle:
            self.velrew_weight = 1.0
            self.stop_velocity_reward = 40.0  # stop giving velocity reward after travelling 40 meters
            self.randomize_initial_state = False
            self.height_penalty = 0.25  # penalize torso to be too high
            self.noisy_input = False


        if self.learn_backflip:
            self.velrew_weight = 0.0
            self.angvel_rew = 1.0
            self.height_threshold_low = 0.26 * self.robot_skeleton.bodynodes[2].com()[1]
            self.rot_threshold = 100000
            self.noisy_input = False

        if self.learn_jump:
            self.velrew_weight = 0.0
            self.height_penalty = -2.0
            self.terminate_for_not_moving = None
            self.dart_world.skeletons[0].bodynodes[1].set_friction_coeff(5.0)
            self.dart_world.skeletons[0].bodynodes[2].set_friction_coeff(5.0)

        if self.learn_getup:
            self.terminate_for_not_moving = None
            self.height_threshold_low = 0.0
            self.rot_threshold = 100000

            self.bodynode_coms = [bn.C for bn in self.robot_skeleton.bodynodes[2:]]
            for i in range(len(self.bodynode_coms)-1, -1, -1):
                self.bodynode_coms[i][0] -= self.bodynode_coms[0][0]

        if self.learn_goto is not None:
            self.terminate_for_not_moving = None


        self.Kp = np.diagflat([0.0] * 3 + [2000.0] * (self.robot_skeleton.ndofs - 3))
        self.Kd = np.diagflat([0.0] * 3 + [20.0] * (self.robot_skeleton.ndofs - 3))

        self.terminator_net = None

        # self.param_manager.set_simulator_parameters([0.9])

        # self.robot_skeleton.joints[3].set_damping_coefficient(0, 1.0)
        # self.robot_skeleton.joints[4].set_damping_coefficient(0, 1.0)
        # self.robot_skeleton.joints[5].set_damping_coefficient(0, 1.0)
        # self.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(0.5)
        # self.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)

        if self.pid_controller is not None:
            self.joint_lower_lim = np.array(self.robot_skeleton.q_lower)[3:]
            self.joint_upper_lim = np.array(self.robot_skeleton.q_upper)[3:]
            expand = (self.joint_upper_lim - self.joint_lower_lim) * 0.2
            self.joint_lower_lim -= expand
            self.joint_upper_lim += expand

        if self.butterworth_filter:
            self.action_filter = ActionFilter(self.act_dim, 3, int(1.0/self.dt), low_cutoff=0.0, high_cutoff=8.0)

        utils.EzPickle.__init__(self)

    def resample_task(self, iter_num=None):
        world_selection = 0  # np.random.randint(len(self.dart_worlds))
        self.dart_world = self.dart_worlds[world_selection]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        self.resample_MP = False

        self.param_manager.resample_parameters()

        ##### MMHACK
        # hack_pm = np.random.choice([0.0, 0.5])
        # self.param_manager.set_simulator_parameters([hack_pm])
        # self.velrew_weight = 1.0
        # if hack_pm < 0.1:
        #     self.velrew_weight = -1.0
        # print(hack_pm)
        #####

        self.current_param = self.param_manager.get_simulator_parameters()

        # self.velrew_weight = np.random.choice([-1.0, 1.0])
        # if iter_num is not None:
        #     if iter_num % 2 == 0:
        #         self.velrew_weight = 1.0
        #     else:
        #         self.velrew_weight = -1.0

        obstacle_height = np.random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        self.obs_height = obstacle_height

        if not self.test_jump_obstacle:
            obstacle_height = -1.0

        for body in self.dart_world.skeletons[1].bodynodes:
            for shapenode in body.shapenodes:
                shapenode.set_offset([self.obstacle_x_offset, obstacle_height, 0])

        return np.array(self.current_param), self.velrew_weight, world_selection, obstacle_height

    def set_task(self, task_params):
        self.dart_world = self.dart_worlds[task_params[2]]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        self.param_manager.set_simulator_parameters(task_params[0])
        self.current_param = self.param_manager.get_simulator_parameters()
        self.velrew_weight = task_params[1]

        for body in self.dart_world.skeletons[1].bodynodes:
            for shapenode in body.shapenodes:
                shapenode.set_offset([self.obstacle_x_offset, task_params[3], 0])
        self.obs_height = task_params[3]

    def pad_action(self, a):
        full_ac = np.zeros(len(self.robot_skeleton.q))
        full_ac[3:] = a
        return full_ac

    def unpad_action(self, a):
        return a[3:]

    def spd(self, target_q, target_dq):
        invM = np.linalg.inv(self.robot_skeleton.M + self.Kd * self.sim_dt)
        p = -self.Kp.dot(self.robot_skeleton.q + (
                    self.robot_skeleton.dq - np.concatenate([[0.0] * 3, target_dq])) * self.sim_dt - np.concatenate(
            [[0.0] * 3, target_q]))
        d = -self.Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.sim_dt
        return tau[3:]

    def track_pose_spd(self, pose, vel):
        tau = self.spd(pose, vel)
        tau = np.concatenate([[0.0] * 3, tau])
        self.robot_skeleton.set_forces(tau)
        self.dart_world.step()

    def do_simulation(self, tau, n_frames):
        if self.pid_controller is not None:
            target_angles = np.copy(tau[3:])#(np.copy(tau[3:]) + 1.0) / 2.0 * (self.joint_upper_lim - self.joint_lower_lim) + self.joint_lower_lim
        average_torque = []
        for _ in range(n_frames):
            if self.pid_controller is not None:
                jpos = np.array(self.robot_skeleton.q)[3:]
                jvel = np.array(self.robot_skeleton.dq)[3:]

                torque = self.pid_controller[0] * (target_angles - jpos) - self.pid_controller[1] * jvel
                clipped_torque = np.clip(torque, self.torque_limit[0], self.torque_limit[1])

                tau = np.concatenate([[0.0] * 3, clipped_torque])
                average_torque.append(clipped_torque)

            if len(self.short_perturb_params) > 0:
                if self.cur_step * self.dt > self.short_perturb_params[0] and \
                        self.cur_step * self.dt < self.short_perturb_params[1]:
                    self.robot_skeleton.bodynodes[2].add_ext_force(self.short_perturb_params[2])

            if self.learnable_perturbation:  # if learn to perturb
                for bid, pert_param in enumerate(self.learnable_perturbation_list):
                    force_dir = self.learnable_perturbation_act[bid * 6: bid * 6 + 3]
                    torque_dir = self.learnable_perturbation_act[bid * 6 + 3: bid * 6 + 6]
                    if np.all(force_dir == 0):
                        pert_force = np.zeros(3)
                    else:
                        pert_force = pert_param[1] * force_dir / np.linalg.norm(force_dir)
                    if np.all(torque_dir == 0):
                        pert_torque = np.zeros(3)
                    else:
                        pert_torque = pert_param[2] * torque_dir / np.linalg.norm(torque_dir)
                    self.robot_skeleton.bodynode(pert_param[0]).add_ext_force(pert_force)
                    self.robot_skeleton.bodynode(pert_param[0]).add_ext_torque(pert_torque)

            if self.use_spline_action:
                self.robot_skeleton.set_forces(tau[_])
            else:
                self.robot_skeleton.set_forces(tau)
            self.joint_works += np.abs(tau * np.array(self.robot_skeleton.dq) * self.sim_dt)
            self.dart_world.step()
        if self.pid_controller is not None:
            self.average_torque = np.mean(average_torque, axis=0)

    def advance(self, a):
        if self.actuator_nonlinearity:
            a = np.tanh(self.actuator_nonlin_coef * a)
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay - 1]

        if self.butterworth_filter:
            a = self.action_filter.filter_action(a)

        # else:
        #     print("1")

        self.posbefore = self.robot_skeleton.q[0]
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        if self.use_spline_action:
            self.spline.set_parameters(a)
            spline_actions = self.spline.get_interpolated_points(int(self.frame_skip / (self.spline_node_num - 1)))
            tau = np.zeros((self.frame_skip, self.robot_skeleton.ndofs))
            tau[:, 3:] = spline_actions * 200
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
        '''if self.cur_step * self.dt > self.short_perturb_params[0] and \
                self.cur_step * self.dt < self.short_perturb_params[1] + 2: # allow 2 seconds to recover
            self.height_threshold_low = 0.0
            self.rot_threshold = 10
        else:
            self.height_threshold_low = 0.56 * self.initial_coms[2][1]
            self.rot_threshold = 0.4'''

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

        # if self.terminator_net is not None and self.cur_step % 20 == 0 and self.cur_step > 1:
        #     s = self.state_vector()
        #     pred = [self.terminator_net.predict(s, use_dropout=True) for i in range(10)]
        #     if np.std(pred) < 0.3 and np.mean(pred) > 0.8:
        #         done = True
        if self.learn_getup:
            if self.cur_step > 300:
                done = True
        if self.learn_alive:
            done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (
                    np.abs(self.robot_skeleton.dq) < 100).all() \
                    and not self.fall_on_ground)

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

        if self.learn_goto is not None:
            reward = 0
            # if self.terminated():
            #     reward = -np.sum(np.square(self.robot_skeleton.q[0:3] - self.learn_goto)) * 500
            pre_rew = -np.sum(np.square(self.q_before[0:3] - self.learn_goto)) * 10
            after_rew = -np.sum(np.square(self.robot_skeleton.q[0:3] - self.learn_goto)) * 10
            reward = after_rew - pre_rew - 0.5
            # if self.terminated():
            #     print(np.array(self.robot_skeleton.q)[0:3], self.learn_goto)

        reward -= np.clip(self.robot_skeleton.dq[2], -self.angvel_clip, self.angvel_clip) * self.angvel_rew
        if posafter > self.stop_velocity_reward:
            reward = 0
        reward += self.alive_bonus * step_skip

        if self.pid_controller is None:
            if not self.void_area:
                reward -= self.energy_penalty * np.square(a).sum()
        else:
            reward -= self.energy_penalty * np.square(self.average_torque / 200.0).sum()

        reward -= 5e-1 * joint_limit_penalty
        reward -= np.abs(self.robot_skeleton.q[2])
        reward -= self.height_penalty * np.clip(self.robot_skeleton.dq[1], 0.0, 1e10)

        reward = np.clip(reward, -np.inf, self.reward_clipping)

        if self.terminator_net is not None:
            s = self.state_vector()
            reward -= self.terminator_net.predict(s, use_dropout=False)[0][0]

        if self.staged_reward:
            if abs(self.robot_skeleton.q[2]) > 0.2 or self.robot_skeleton.bodynodes[2].com()[1] < 0.85:
                reward = 0.3 - 0.5*np.sum(np.abs(np.array(self.robot_skeleton.q)[1:]))

        if self.learn_getup:
            cur_bodynode_coms = [bn.C for bn in self.robot_skeleton.bodynodes[2:]]
            for i in range(len(cur_bodynode_coms) - 1, -1, -1):
                cur_bodynode_coms[i][0] -= cur_bodynode_coms[0][0]

            reward = 3.0
            for b in range(len(cur_bodynode_coms)):
                reward -= np.linalg.norm(cur_bodynode_coms[b] - self.bodynode_coms[b])

        vrew = (posafter - self.posbefore) / self.dt * self.velrew_weight
        abrew = self.alive_bonus * step_skip
        jlimrew = 5e-1 * joint_limit_penalty
        qrew = np.abs(self.robot_skeleton.q[2])
        hrew = self.height_penalty * np.clip(self.robot_skeleton.dq[1], 0.0, 1e10)
        # if abs(reward) > 10:
        #     print(vrew, abrew, jlimrew, qrew, hrew, reward)

        # if self.quantized_observation:
        #     if self.quantization_param[2] < 0.5:
        #         reward += 0.2
        #     if self.current_quant_budget == 0:
        #         reward -= 1.0
            # reward += self.current_quant_budget / self.max_quant_budget * 2.0 #0.0 * (np.exp(-7*self.quantization_param[2])-np.exp(-7))

        if self.learn_alive:
            reward = 1.0

        return reward

    def step(self, a):
        prev_pose = np.array(self.robot_skeleton.q)
        if self.pid_controller is None:
            action_bound_violation = np.sum(np.clip(np.abs(a) - 1, 0, 10000))
        else:
            action_bound_violation = 0.0
        if self.pseudo_lstm_dim > 0:
            self.hidden_states = a[self.act_dim - self.pseudo_lstm_dim * 2:]
            a = a[0:self.act_dim - self.pseudo_lstm_dim * 2]
        self.action_filter_cache.append(a)
        if len(self.action_filter_cache) > self.action_filtering:
            self.action_filter_cache.pop(0)
        if self.action_filtering > 0 and self.action_filter_in_env:
            a = np.mean(self.action_filter_cache, axis=0)

        if self.quantized_observation:
            self.quantization_param[2] = 0 if a[-1] < 0.0 else 1
            if self.quantization_param[2] > 0.5:
                self.current_quant_budget = max(self.current_quant_budget - 1, 0)
            if self.current_quant_budget == 0:
                self.quantization_param[2] = 0
            # self.quantization_param[2] = np.clip(a[-1], -1, 1) * 0.5 + 0.5

        if self.vibrating_ground:
            # self.dart_world.skeletons[0].joints[0].set_rest_position(0, self.ground_vib_params[0] * np.sin(
            #     2 * np.pi * self.ground_vib_params[1] * self.cur_step * self.dt))
            self.dart_world.skeletons[0].bodynodes[0].shapenodes[0].set_offset([0,
                                                                                self.ground_vib_params[0] * np.sin(2 * np.pi * self.ground_vib_params[1] * self.cur_step * self.dt), 0])
            self.dart_world.skeletons[0].bodynodes[0].shapenodes[1].set_offset([0,
                                                                                self.ground_vib_params[0] * np.sin(2 * np.pi * self.ground_vib_params[1] * self.cur_step * self.dt), 0])

        if self.action_bound_model is not None:
            pred_bound = self.action_bound_model.predict(self._get_obs(False))[0]
            in_a = np.copy(a)
            up_bound = pred_bound[::2]
            low_bound = pred_bound[1::2]
            mid = 0.5 * (up_bound + low_bound)
            up_bound[up_bound - low_bound < 0.05] = mid[up_bound - low_bound < 0.05] + 0.05
            low_bound[up_bound - low_bound < 0.05] = mid[up_bound - low_bound < 0.05] - 0.05
            a = in_a * (up_bound - low_bound) + low_bound

        self.t += self.dt
        self.pre_advance()
        self.advance(a)
        reward = self.reward_func(a)
        if not self.void_area:
            reward -= self.action_bound_penalty * action_bound_violation

        # MMHACK: modify friction in the middle
        # if self.resample_MP:
        #     assert(0)
        # if self.cur_step > 400:
        #     self.param_manager.set_simulator_parameters([0.25])
        # else:
        #     self.param_manager.set_simulator_parameters([0.6])
        # if self.cur_step > 200:
        #     mult = 0.94
        #     self.action_scale = np.array([200.0 * mult, 200.0 * mult, 200.0 * mult, 1.0])

        done = self.terminated() or self.cur_step >= 999

        if self.sparse_reward:
            if done:
                reward = self.accumulated_reward
            else:
                self.accumulated_reward += reward
                reward = 0
        else:
            if done:
                reward = 0

        ob = self._get_obs()

        self.cur_step += 1

        envinfo = {}

        contacts = self.dart_world.collision_result.contacts
        for contact in contacts:
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
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
        if not self.randomize_obstacles:
            vertical_range = [-5, -5]
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
        # state = np.array(self.robot_skeleton.q[1:])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.two_pose_input:
            self.pose_history.append(np.copy(self.robot_skeleton.q))
            state = np.concatenate([
                self.pose_history[-1][1:],
                self.pose_history[-2][1:] - self.pose_history[-1][1:],
            ])
            state[0] = self.robot_skeleton.bodynodes[2].com()[1]
            state[5] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.action_filtering > 0 and self.action_filter_inobs:
            state = np.concatenate([state] + self.action_filter_cache)

        if self.test_jump_obstacle:
            state = np.concatenate([state, [self.robot_skeleton.q[0] - 3.5]])
            if self.input_obs_height:
                state = np.concatenate([state, [self.obs_height]])

        if self.noisy_input:
            state = state + np.random.normal(0, .05, len(state))

        if self.quantized_observation:
            # quantize_size = int(self.quantization_param[2] * (self.quantization_param[1] - self.quantization_param[0]) + self.quantization_param[0])
            # # quantize_size = 5
            # quantize_interv = 1.0 / (quantize_size - 1)
            #
            # state = np.clip(state, self.obs_ranges[:, 0], self.obs_ranges[:, 1] + 0.001)
            # normalized_state = (state - self.obs_ranges[:, 0]) / (self.obs_ranges[:, 1] - self.obs_ranges[:, 0])
            # quantized_state = np.floor(normalized_state / quantize_interv) * quantize_interv
            # state = quantized_state * (self.obs_ranges[:, 1] - self.obs_ranges[:, 0]) + self.obs_ranges[:, 0]
            # state = np.concatenate([state, [self.quantization_param[2], self.current_quant_budget / self.max_quant_budget]])
            if self.quantization_param[2] < 0.5:
                state = self.observation_buffer[-1][0:11]
                cur_choice = 0
            else:
                cur_choice = 1
            if cur_choice == self.last_choice:
                self.choice_duration += 1
            else:
                self.last_choice = cur_choice
                self.choice_duration = 0
            state = np.concatenate([state, [cur_choice, self.choice_duration * 0.1, self.current_quant_budget / self.max_quant_budget]])


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

        if self.periodic_noise:
            final_obs += np.random.randn(len(final_obs)) * self.periodic_noise_params[0] * (
                        np.sin(2 * np.pi * self.periodic_noise_params[1] * self.cur_step * self.dt) + 1)

        if self.pseudo_lstm_dim > 0:
            final_obs = np.concatenate([final_obs, self.hidden_states])

        if self.diff_obs:
            single_obs = np.copy(final_obs)
            for i in range(len(single_obs) - 1):
                final_obs = np.concatenate([final_obs, single_obs - np.roll(single_obs, i + 1)])

        if self.append_zeros > 0:
            final_obs = np.concatenate([final_obs, 500 * np.random.random(self.append_zeros)])

        if self.obs_projection_model:
            final_obs = self.obs_projection_model(final_obs)

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

        # qpos[2] = np.random.uniform(-0.3, 0.3)
        # qvel[0] = np.random.uniform(-1.0, 1.0)
        # qvel[2] = np.random.uniform(-1.0, 1.0)

        if self.learn_getup:
            # qpos[2] = np.random.uniform(-0.7, 0.7)
            if np.random.random() > 0.5:
                qpos[2] = np.random.uniform(0.6, 1.2)
            else:
                qpos[2] = np.random.uniform(-0.6, -1.2)
            qpos[5] = np.random.uniform(0.0, 1.0)
            qpos[1] = -1.0
            self.set_state(qpos, qvel)
            while self.robot_skeleton.bodynodes[-1].C[1] < 0.1:
                qpos[1] += 0.01
                self.set_state(qpos, qvel)

        if self.learn_alive:
            if np.random.random() > 0.5:
                qpos[2] = np.random.uniform(0.6, 0.7)
            else:
                qpos[2] = np.random.uniform(-0.6, -0.7)
            qpos[5] = np.random.uniform(0.0, 1.0)
            self.set_state(qpos, qvel)
            while self.robot_skeleton.bodynodes[-1].C[1] < 0.1:
                qpos[1] += 0.05
                self.set_state(qpos, qvel)

        if self.quantized_observation:
            self.current_quant_budget = self.max_quant_budget
            self.quantization_param = [3, 50, 1.0]  # min, max, current
            self.last_choice = 1
            self.choice_duration = 0

        self.set_state(qpos, qvel)

        if self.resample_MP:
            self.param_manager.resample_parameters()
        self.current_param = self.param_manager.get_simulator_parameters()

        self.observation_buffer = []
        self.action_buffer = []

        self.action_filter_cache = []
        if self.action_filtering > 0:
            for i in range(self.action_filtering):
                self.action_filter_cache.append(np.zeros(len(self.action_scale)))

        if self.two_pose_input:
            self.pose_history = [np.copy(self.robot_skeleton.q)]

        self.history_buffers = []

        if self.staged_reward:
            valid_init = False
            while not valid_init:
                self.set_state(qpos, qvel)
                for i in range(40):
                    self.robot_skeleton.set_forces(np.concatenate([[0, 0, 0], np.random.uniform(-200, 200, 3)]))
                    self.dart_world.step()
                done = self.terminated()
                if not done:
                    valid_init = True

        state = self._get_obs(update_buffer=True)

        # setup obstacle
        horizontal_range = [0.5, 0.5]
        vertical_range = [-0.01, 0.1]
        if not self.randomize_obstacles:
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

        if self.vibrating_ground:
            self.ground_vib_params[0] = np.random.random() * 0.14

        self.learnable_perturbation_act = np.zeros(len(self.learnable_perturbation_list) * 6)

        if self.pseudo_lstm_dim > 0:
            self.hidden_states = np.zeros(self.pseudo_lstm_dim * 2)

        self.accumulated_reward = 0

        self.joint_works = np.zeros(len(self.robot_skeleton.dq))

        if self.butterworth_filter:
            self.action_filter.reset_filter()

        return state

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

    def true_reward(self):
        return self.robot_skeleton.q[0]  # position in x direction