import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *
from gym.envs.dart.sub_tasks import *
import copy

import joblib, os

class DartAntEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*8,[-1.0]*8])
        self.action_scale = 150.0
        self.train_UP = False
        self.noisy_input = False
        obs_dim = 27

        self.training_mode = True

        self.velrew_weight = 3.0
        self.resample_MP = False  # whether to resample the model paraeters
        self.param_manager = antParamManager(self)

        self.random_direction = True

        self.bridge_puzzle = True

        self.t = 0

        self.total_dist = []

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history

        if self.random_direction:
            obs_dim += 2

        obs_perm_base = np.array(
            [0.0001, -1, 2, -3, -4,   -7,8, -5,6, -11,12, -9,10, 13,14,-15,16,-17,-18, -21,22, -19,20, -25,26, -23,24])
        act_perm_base = np.array([-2,3, -0.0001,1, -6,7, -4,5])
        self.obs_perm = np.copy(obs_perm_base)

        for i in range(self.include_obs_history-1):
            self.obs_perm = np.concatenate([self.obs_perm, np.sign(obs_perm_base) * (np.abs(obs_perm_base) + len(self.obs_perm))])
        for i in range(self.include_act_history):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(act_perm_base) * (np.abs(act_perm_base) + len(self.obs_perm))])
        self.act_perm = np.array([-2,3, -0.0001,1, -6,7, -4,5])

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)
            self.obs_perm = np.concatenate([self.obs_perm, np.arange(int(len(self.obs_perm)),
                                                int(len(self.obs_perm)+len(self.param_manager.activated_param)))])

        if self.bridge_puzzle:
            dart_env.DartEnv.__init__(self, ['bridge_puzzle_ant.skel'], 5, obs_dim, self.control_bounds, disableViewer=True, dt=0.01)
        else:
            dart_env.DartEnv.__init__(self, ['ant.skel'], 5, obs_dim, self.control_bounds,
                                      disableViewer=True, dt=0.01)

        self.initial_local_coms = [np.copy(bn.local_com()) for bn in self.robot_skeleton.bodynodes]

        #self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_worlds[0].set_collision_detector(3)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 0
        self.act_delay = 0

        self.tilt_x = 0
        self.tilt_z = 0
        self.current_param = self.param_manager.get_simulator_parameters()

        #self.param_manager.set_simulator_parameters(self.current_param)

        #print('sim parameters: ', self.param_manager.get_simulator_parameters())

        # data structure for actuation modeling
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]

        self.targets = [np.array([0.0, 0.5, 0.0]), np.array([1.0, 0.5, 0.0]), np.array([2.0, 0.5, 0.0]),
                        np.array([3.0, 0.5, 0.0]), np.array([4.0, 0.5, 0.0]),
                        np.array([4.7, 0.5, -0.7]), np.array([5.4, 0.5, -1.4]), np.array([6.1, 0.5, -2.1]),
                        np.array([6.8, 0.5, -2.8]), np.array([7.5, 0.5, -3.5]),
                        np.array([8.33, 0.5, -2.8]), np.array([9.16, 0.5, -2.1]), np.array([10.0, 0.5, -1.4]),
                        np.array([10.83, 0.5, -0.7]), np.array([11.66, 0.5, 0.0]), np.array([12.5, 0.5, 0.7]),
                        np.array([13.33, 0.5, 0.0]), np.array([14.16, 0.5, -0.7]), np.array([15.0, 0.5, -1.4]),
                        np.array([15.83, 0.5, -2.1]), np.array([16.66, 0.5, -2.8]), np.array([17.5, 0.5, -3.5]),
                        np.array([18.33, 0.5, -2.8]), np.array([19.16, 0.5, -2.1]), np.array([20.0, 0.5, -1.4]),
                        np.array([20.83, 0.5, -0.7]), np.array([21.66, 0.5, 0.0]), np.array([22.5, 0.5, 0.7]),
                        np.array([23.0, 0.5, 0.7]), np.array([23.5, 0.5, 0.7])]

        utils.EzPickle.__init__(self)

    def pad_action(self, a):
        full_ac = np.zeros(len(self.robot_skeleton.q))
        full_ac[6:] = a
        return full_ac

    def unpad_action(self, a):
        return a[6:]

    def advance(self, a):
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay-1]

        self.posbefore = self.robot_skeleton.q[0]
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale
        self.do_simulation(tau, self.frame_skip)

    def post_advance(self):
        self.dart_world.check_collision()

    def terminated(self):
        fall_on_ground = False
        permitted_contact_bodies = [self.dart_world.skeletons[0].bodynodes[0], self.robot_skeleton.bodynodes[4],
                                    self.robot_skeleton.bodynodes[7], self.robot_skeleton.bodynodes[10],
                                    self.robot_skeleton.bodynodes[13]]
        contacts = self.dart_world.collision_result.contacts
        for contact in contacts:
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
            if contact.bodynode1 not in permitted_contact_bodies or contact.bodynode2 not in permitted_contact_bodies:
                fall_on_ground = True

        s = self.state_vector()
        height = self.robot_skeleton.bodynodes[2].com()[1]
        done = not (np.isfinite(s).all() and (np.abs(s) < 200).all()\
            and (height > 0.3) and height < 1.0)
        return done# or fall_on_ground

    def pre_advance(self):
        self.posbefore = self.robot_skeleton.q[0]
        self.xyposbefore = np.array([self.robot_skeleton.q[0], self.robot_skeleton.q[2]])

    def reward_func(self, a, step_skip=1):
        posafter = self.robot_skeleton.q[0]
        self.xyposafter = np.array([self.robot_skeleton.q[0], self.robot_skeleton.q[2]])
        deviation = self.robot_skeleton.q[2]
        self.alive_bonus = 1.0
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight

        if self.random_direction:
            vel = (self.xyposafter - self.xyposbefore) / self.dt
            reward = np.sum(self.goal_direction * vel) * self.velrew_weight
            reward -= np.linalg.norm(vel - np.sum(self.goal_direction * vel) * self.goal_direction) * self.velrew_weight * 0.3  # perpendicular component

        reward += self.alive_bonus * step_skip
        reward -= 0.1 * np.square(a).sum()
        if not self.random_direction:
            reward -= np.abs(deviation) * 3

        s = self.state_vector()
        if not(np.isfinite(s).all() and (np.abs(s) < 200).all()):
            reward = 0

        return reward

    def step(self, a):
        self.t += self.dt
        self.pre_advance()
        self.advance(a)
        reward = self.reward_func(a)

        done = self.terminated()

        ob = self._get_obs()

        self.cur_step += 1

        envinfo = {}

        if np.random.random() < 0.01 and self.random_direction:
            angle = np.random.uniform(0, 2 * np.pi)
            self.goal_direction = np.array([np.sin(angle), np.cos(angle)])

        if self.bridge_puzzle:
            vec = self.robot_skeleton.C - self.targets[self.current_target_id]

            if np.linalg.norm(vec) < 0.5:
                reward += 50
                self.current_target_id += 1
                if self.current_target_id >= len(self.targets):
                    self.current_target_id -= 1
                self.dart_world.skeletons[-2].q = self.targets[self.current_target_id]

        if self.training_mode:
            if np.linalg.norm(self.robot_skeleton.dC) < 1.0:
                self.slow_vel_count += 1
            else:
                self.slow_vel_count = 0
            if self.slow_vel_count > 20:
                done = True
        # if self.cur_step > 30 and np.linalg.norm(self.robot_skeleton.dC) < 0.1: # if moving too slowly
        #     done = True

        return ob, reward, done, envinfo

    def _get_obs(self, update_buffer = True):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])
        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        if self.random_direction:
            state[1] = 0.0

        state[3] = state[3] % (2 * np.pi)

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

        if self.random_direction:
            final_obs = np.concatenate([final_obs, self.goal_direction])

            # dir = np.array([0.0, 1.0])
            # final_obs = np.concatenate([final_obs, dir/np.linalg.norm(dir)])

        return final_obs

    def reset_model(self):
        for world in self.dart_worlds:
            world.reset()
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]
        init_q = np.array([0, -0.2, 0, 0,0,0, 0,-1, 0,-1, 0,1, 0,1])
        qpos = init_q + self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
        # if self.training_mode:
        #     qpos[4] = np.random.uniform(0, np.pi*2)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)
        if self.resample_MP:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()

        self.observation_buffer = []
        self.action_buffer = []

        if self.random_direction:
            angle = np.random.uniform(0, 2*np.pi)
            self.goal_direction = np.array([np.sin(angle), np.cos(angle)])

        state = self._get_obs(update_buffer = True)

        self.cur_step = 0

        self.height_threshold_low = 0.56*self.robot_skeleton.bodynodes[2].com()[1]
        self.t = 0

        self.fall_on_ground = False

        self.slow_vel_count = 0

        self.current_target_id = 0

        if self.bridge_puzzle:
            self.dart_world.skeletons[-2].q = self.targets[0]

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5

        if self.random_direction:
            self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[2] = -10.5
            self._get_viewer().scene.tb._set_theta(-60)


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
