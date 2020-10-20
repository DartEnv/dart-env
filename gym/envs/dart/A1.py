import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *
import copy

import joblib, os
from pydart2.utils.transformations import quaternion_from_matrix, euler_from_matrix, euler_from_quaternion

DEBUG = False

DEFAULT_POSE = [0.0, 0.67, -1.25] * 4
DEFAULT_HEIGHT = 0.35

class DartA1Env(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*12,[-1.0]*12])

        self.action_bounds = [np.array([0.802851455917, 1.67, -0.75] * 4),
                              np.array([-0.802851455917, -0.33, -1.75] * 4)]

        self.controller_p_gains = np.array([100.0] * 12)
        self.controller_d_gains = np.array([1.0, 2.0, 2.0] * 4)

        self.torque_limits = np.array([20.0] * 12)

        self.max_target_change = 0.2

        self.with_reality_gap = False

        self.control_interval = 0.01 # 100 Hz
        self.sim_timestep = 0.002 # simulate at 500Hz

        self.train_UP = False
        self.noisy_input = True
        self.randomize_initial_state = True
        obs_dim = 6 + 3 + 1 + 12 * 2 # roll, pitch, yaw, dr, dp, dy, xyz velocity, y deviation, joint states/velocities

        # Reward related
        self.velrew_weight = 3.0
        self.velocity_clip = 2.5
        self.alive_bonus = 1.0
        self.target_diff_penalty = 0.025
        self.deviation_penalty = 5.0
        self.hip_penalty = 1.0
        self.joint_velocity_limit = 1000
        self.joint_velocity_penalty = 0.005

        self.resample_MP = True  # whether to resample the model paraeters

        self.max_actuator_power = None

        self.soft_ground = False

        self.param_manager = a1ParamManager(self)

        if self.train_UP:
            obs_dim += len(self.param_manager.param_dim)

        self.t = 0

        self.total_dist = []

        self.include_obs_history = 1
        self.include_act_history = 1
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history

        obs_perm_base = np.array(
            [-0.0001,1,-2, -3,4,-5, 6,-7,8, -9, -13,14,15, -10,11,12, -19,20,21, -16,17,18,
             -25,26,27, -22,23,24, -31,32,33, -28,29,30])
        act_perm_base = np.array([-3, 4, 5, -0.0001, 1, 2, -9,10,11, -6,7,8])
        self.obs_perm = np.copy(obs_perm_base)

        for i in range(self.include_obs_history - 1):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(obs_perm_base) * (np.abs(obs_perm_base) + len(self.obs_perm))])
        for i in range(self.include_act_history):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(act_perm_base) * (np.abs(act_perm_base) + len(self.obs_perm))])

        if self.train_UP:
            obs_dim += self.param_manager.param_dim
            self.obs_perm = np.concatenate([self.obs_perm, np.arange(int(len(self.obs_perm)),
                                                                     int(len(self.obs_perm) +
                                                                         self.param_manager.param_dim))])

        self.act_perm = np.array(act_perm_base)

        model_file_list = ['a1/ground1.urdf', 'a1/a1.URDF']

        dart_env.DartEnv.__init__(self, model_file_list, int(self.control_interval / self.sim_timestep), obs_dim,
                                  self.control_bounds, dt=self.sim_timestep, disableViewer=True,
                                  action_type="continuous")

        self.dart_world.set_gravity([0, 0, -9.81])

        self.default_mass = [bn.mass() for bn in self.robot_skeleton.bodynodes]
        self.initial_local_coms = [np.copy(bn.local_com()) for bn in self.robot_skeleton.bodynodes]
        self.default_controller_p_gains = np.array(self.controller_p_gains)

        self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_worlds[0].set_collision_detector(0)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 0
        self.act_delay = 0

        if self.with_reality_gap:
            self.obs_delay = 0
            self.max_actuator_power = 50


        print('sim parameters: ', self.param_manager.get_simulator_parameters())
        self.current_param = self.param_manager.get_simulator_parameters()

        self.add_perturbation = True
        self.perturbation_parameters = [0.05, 20, 1, 20]  # probability, magnitude, bodyid, duration
        self.perturb_offset = [0.0, 0.0, 0.0]

        if self.with_reality_gap:
            self.add_perturbation = False

        self.dart_worlds[0].set_collision_detector(3)

        utils.EzPickle.__init__(self)

    def resample_task(self):
        pass

    def set_task(self, task_params):
        pass

    def do_simulation(self, target_pose, n_frames):
        if self.add_perturbation:
            if self.perturbation_duration == 0:
                self.perturb_force *= 0
                if np.random.random() < self.perturbation_parameters[0]:
                    axis_rand = np.random.randint(0, 2, 1)[0]
                    direction_rand = np.random.randint(0, 2, 1)[0] * 2 - 1
                    self.perturb_force[axis_rand] = direction_rand * self.perturbation_parameters[1]
                    self.perturbation_duration = self.perturbation_parameters[3]
                    self.perturb_offset = np.array([np.random.uniform(-0.5, 0.5), 0.0, 0.0])
            else:
                self.perturbation_duration -= 1

        for _ in range(n_frames):
            if self.add_perturbation:
                self.robot_skeleton.bodynodes[self.perturbation_parameters[2]].add_ext_force(self.perturb_force, _offset = self.perturb_offset)

            torque = self.compute_torque(target_pose)
            self.last_torque = np.array(torque)

            self.robot_skeleton.set_forces(np.concatenate([np.zeros(6), torque]))
            self.dart_world.step()

    def compute_torque(self, target_pose):
        current_joint_pose = np.array(self.robot_skeleton.q)[6:]
        current_joint_velocity = np.array(self.robot_skeleton.dq)[6:]

        torque = self.controller_p_gains * (
                    target_pose - current_joint_pose) - self.controller_d_gains * current_joint_velocity
        if self.max_actuator_power is not None:
            max_torque = np.abs(self.max_actuator_power / current_joint_velocity)
            torque = np.clip(torque, -max_torque, max_torque)
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)

        # print(current_joint_pose)
        # print(current_joint_velocity)
        # print(target_pose)
        # print(torque)
        # import pdb; pdb.set_trace()
        # print("====")

        return torque

    def advance(self, a):
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay-1]

        self.posbefore = self.robot_skeleton.q[3]
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        target_pose = (clamped_control * 0.5 + 0.5) * (self.action_bounds[0] - self.action_bounds[1]) + self.action_bounds[1]

        self.target_diff = np.abs(target_pose - self.last_target_pose)

        target_pose = np.clip(target_pose, self.last_target_pose - self.max_target_change,
                              self.last_target_pose + self.max_target_change)

        self.last_target_pose = np.copy(target_pose)

        self.do_simulation(target_pose, self.frame_skip)


    def post_advance(self):
        self.dart_world.check_collision()

    def check_fall_on_ground(self):
        fog = False
        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        permitted_contact_bodies = [self.robot_skeleton.bodynodes[6], self.robot_skeleton.bodynodes[16],
                                    self.robot_skeleton.bodynodes[11], self.robot_skeleton.bodynodes[21]]
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
            if contact.bodynode1 not in permitted_contact_bodies and contact.bodynode2 not in permitted_contact_bodies:
                fog = True
        return fog

    def terminated(self):
        s = self.state_vector()

        if s[5] < 0.1:
            if DEBUG:
                print("COM height too low")
            return True

        if np.abs(s[0]) > 0.4:
            if DEBUG:
                print("Roll exceed limit")
            return True

        if np.abs(s[1]) > 0.6:
            if DEBUG:
                print("Pitch exceed limit")
            return True

        if self.check_fall_on_ground():
            if DEBUG:
                print("Falling on ground")
            return True

        if np.abs(s[2]) > 1.0:
            if DEBUG:
                print("Yaw exceed limit")
            return True

        if np.abs(s[4]) > 2.0:
            if DEBUG:
                print("Lateral distance exceed limit")
            return True

        joint_velocities = np.array(self.robot_skeleton.dq)[6:]
        if np.any(np.abs(joint_velocities) > self.joint_velocity_limit):
            if DEBUG:
                print("Joint velocity limit exceeded")
            return True

        return False

    def pre_advance(self):
        self.posbefore = self.robot_skeleton.q[3]

    def reward_func(self, a, step_skip=1):
        posafter = self.robot_skeleton.q[3]
        vel_rew = np.clip((posafter - self.posbefore) / self.dt, -self.velocity_clip,
                         self.velocity_clip) * self.velrew_weight
        alive_rew = self.alive_bonus * step_skip
        energy_rew = - self.target_diff_penalty * np.square(self.target_diff).sum()
        deviation_rew = - np.max([np.abs(self.robot_skeleton.q[4]) - 0.25, 0.0]) * self.deviation_penalty

        hip_angles = np.array(self.robot_skeleton.q)[[6, 9, 12, 15]]
        hip_angles[np.abs(hip_angles) < 0.1] = 0.0
        hip_rew = -np.sum(np.abs(hip_angles)) * self.hip_penalty

        joint_velocities = np.array(self.robot_skeleton.dq)[6:]
        joint_vel_rew = -np.mean(np.square(joint_velocities)) * self.joint_velocity_penalty

        reward = vel_rew + alive_rew + energy_rew + deviation_rew + hip_rew + joint_vel_rew
        # print("vel_rew: {}, alive_rew: {}, energy_rew: {}, deviation_rew: {}, hip_rew: {}".format(vel_rew, alive_rew, energy_rew, deviation_rew, hip_rew))

        return reward

    def step(self, a):
        self.t += self.dt
        self.pre_advance()
        self.advance(a)
        reward = self.reward_func(a)

        done = self.terminated()

        if done:
            reward = 0.0

        ob = self._get_obs()

        self.cur_step += 1

        envinfo = {}

        return ob, reward, done, envinfo

    def _get_obs(self, update_buffer = True):
        q = np.array(self.robot_skeleton.q)
        dq = np.array(self.robot_skeleton.dq)

        dr_dp_dy = self.robot_skeleton.bodynodes[1].com_spatial_velocity()[0:3]
        roll_pitch_yaw = np.array(euler_from_matrix(self.robot_skeleton.bodynodes[1].T[0:3, 0:3], 'sxyz'))

        lin_vel = dq[3:6]
        y_deviation = q[4]
        state =  np.concatenate([
            roll_pitch_yaw, dr_dp_dy, lin_vel, [y_deviation],
            q[6:], dq[6:]
        ])

        if self.train_UP:
            UP = self.param_manager.get_simulator_parameters()
            state = np.concatenate([state, UP])
        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        if update_buffer:
            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay-1-i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0]*0.0])

        final_obs = np.concatenate([final_obs, self.last_target_pose])
        # for i in range(self.include_act_history):
        #     if i < len(self.action_buffer):
        #         final_obs = np.concatenate([final_obs, self.action_buffer[-1-i]])
        #     else:
        #         final_obs = np.concatenate([final_obs, [0.0]*len(self.control_bounds[0])])

        return final_obs

    def reset_model(self):
        for world in self.dart_worlds:
            world.reset()
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]
        qpos = np.array(self.robot_skeleton.q)
        qpos[5] = DEFAULT_HEIGHT
        qpos[6:] = np.array(DEFAULT_POSE)
        qvel = np.array(self.robot_skeleton.dq)

        self.last_target_pose = np.array(DEFAULT_POSE)

        if self.randomize_initial_state and not self.with_reality_gap:
            qpos += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
            qvel += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)


        self.set_state(qpos, qvel)
        if self.resample_MP and not self.with_reality_gap:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()

        self.observation_buffer = []
        self.action_buffer = []

        state = self._get_obs(update_buffer = True)

        self.cur_step = 0

        self.t = 0

        return state

    def viewer_setup(self):
        if not self.disableViewer:
            # self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -2.0
            self._get_viewer().scene.tb.trans[1] = 0.0
            self._get_viewer().scene.tb.theta = 70
            self._get_viewer().scene.tb.phi = 0

        return 0

    def state_vector(self):
        s = np.copy(np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]))
        return s

    def set_state_vector(self, s):
        snew = np.copy(s)
        self.robot_skeleton.q = snew[0:len(self.robot_skeleton.q)]
        self.robot_skeleton.dq = snew[len(self.robot_skeleton.q):]

    def set_sim_parameters(self, pm):
        self.param_manager.set_simulator_parameters(pm)

    def get_sim_parameters(self):
        return self.param_manager.get_simulator_parameters()
