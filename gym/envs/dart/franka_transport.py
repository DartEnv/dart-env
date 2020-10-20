import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *
import copy

import joblib, os
from pydart2.utils.transformations import quaternion_from_matrix, euler_from_matrix, euler_from_quaternion

DEBUG = False

DEFAULT_POSE = np.array([-1.0, 0.0, 0.0, -1.57, 1.57, 3.14, -0.785])

class DartFrankaTransportEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*7,[-1.0]*7])

        self.action_bounds = [np.array([2.0, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
                              np.array([-2.0, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])]

        self.controller_p_gains = np.array([100.0] * 7)
        self.controller_d_gains = np.array([1.0] * 7)

        self.torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

        self.max_target_change = 0.2

        self.control_interval = 0.03 # 100 Hz
        self.sim_timestep = 0.003 # simulate at 500Hz

        self.initial_object_position = np.array([ 0.28762935, -0.599192  ,  0.80200745])
        self.target_object_position = np.array([-0.46762935, -0.579192  ,  0.73200745])

        self.train_UP = False
        self.noisy_input = False
        self.randomize_initial_state = False
        obs_dim = 6 + 6 + 6 + 6  # joint pose velocity, last target pose, object 6dof

        # Reward related
        self.distance_approach_weight = 100.0
        self.alive_bonus = 0.5

        self.resample_MP = False  # whether to resample the model paraeters

        self.max_actuator_power = None

        self.param_manager = frankaParamManager(self)

        if self.train_UP:
            obs_dim += len(self.param_manager.param_dim)

        self.t = 0

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history

        model_file_list = ['franka/ground1.urdf', 'franka/object.urdf', 'franka/object_target.urdf', 'franka/panda_arm_hand_tool.URDF']

        dart_env.DartEnv.__init__(self, model_file_list, int(self.control_interval / self.sim_timestep), obs_dim,
                                  self.control_bounds, dt=self.sim_timestep, disableViewer=True,
                                  action_type="continuous")

        self.object = self.dart_world.skeletons[1]
        self.target_visualization = self.dart_world.skeletons[2]
        self.dart_world.set_gravity([0, 0, -9.81])
        # self.robot_skeleton.set_self_collision_check(True)

        self.default_mass = [bn.mass() for bn in self.robot_skeleton.bodynodes]
        self.initial_local_coms = [np.copy(bn.local_com()) for bn in self.robot_skeleton.bodynodes]
        self.default_controller_p_gains = np.array(self.controller_p_gains)

        self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_worlds[0].set_collision_detector(2)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 0
        self.act_delay = 0


        print('sim parameters: ', self.param_manager.get_simulator_parameters())
        self.current_param = self.param_manager.get_simulator_parameters()

        self.add_perturbation = False
        self.perturbation_parameters = [0.02, 10, -1, 20]  # probability, magnitude, bodyid, duration
        self.perturb_offset = [0.0, 0.0, 0.0]

        self.dart_worlds[0].set_collision_detector(3)

        self.target_visualization.bodynodes[0].shapenodes[0].set_offset(self.target_object_position)

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

            self.robot_skeleton.set_forces(torque)
            self.dart_world.step()

    def _gravity_compensation_torque(self):
        gc_tau = np.zeros(7)
        for body in self.robot_skeleton.bodynodes[1:]:
            m = body.mass()  # Or, simply body.m
            J = body.linear_jacobian(body.local_com())
            gc_tau += J.transpose().dot(-(m * np.array([0.0, 0.0, -9.87])))
        return gc_tau

    def compute_torque(self, target_pose):
        current_joint_pose = np.array(self.robot_skeleton.q)
        current_joint_velocity = np.array(self.robot_skeleton.dq)

        torque = self.controller_p_gains * (
                    target_pose - current_joint_pose) - self.controller_d_gains * current_joint_velocity

        if self.max_actuator_power is not None:
            max_torque = np.abs(self.max_actuator_power / current_joint_velocity)
            torque = np.clip(torque, -max_torque, max_torque)

        torque += self._gravity_compensation_torque()

        torque = np.clip(torque, -self.torque_limits, self.torque_limits)

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


    def terminated(self):
        hand_position = self.robot_skeleton.bodynodes[-1].C
        obj_position = self.object.C

        if obj_position[2] < hand_position[2] - 0.05 or obj_position[2] > hand_position[2] + 0.1:
            if DEBUG:
                print("Object fallen")
            return True

        if hand_position[2] < 0.5:
            if DEBUG:
                print("Hand too low")
            return True

        if np.linalg.norm(hand_position - obj_position) > 0.2:
            if DEBUG:
                print("Hand-object too far")
            return True

        return False

    def pre_advance(self):
        self.obj_position_before = np.array(self.object.C)

    def reward_func(self, a, step_skip=1):
        distance_before = np.linalg.norm(self.obj_position_before - self.target_object_position)
        distance = np.linalg.norm(self.object.C - self.target_object_position)
        distance_rew = np.clip(distance - distance_before, -0.005, 0.005)

        reward = self.alive_bonus - distance_rew * self.distance_approach_weight

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
        object_position = self.object.C

        state = np.concatenate([q, dq, self.last_target_pose, object_position])

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

        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[-1-i]])
            else:
                final_obs = np.concatenate([final_obs, [0.0]*len(self.control_bounds[0])])

        return final_obs

    def reset_model(self):
        for world in self.dart_worlds:
            world.reset()
        qpos = np.copy(DEFAULT_POSE)
        qvel = np.array(self.robot_skeleton.dq)

        self.last_target_pose = np.array(DEFAULT_POSE)

        object_q = np.array([0, 0, 0] + self.initial_object_position.tolist())
        object_dq = np.zeros(6)

        if self.randomize_initial_state:
            qpos += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
            qvel += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
            object_q += self.np_random.uniform(low=-.005, high=.005, size=6)
            object_dq += self.np_random.uniform(low=-.005, high=.005, size=6)

        self.object.q = object_q
        self.object.dq = object_dq

        self.set_state(qpos, qvel)
        if self.resample_MP:
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
            # self.track_skeleton_id = -1
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -2.5
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
