import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *
from gym.envs.dart.sub_tasks import *
import copy

import joblib, os
from pydart2.utils.transformations import quaternion_from_matrix, euler_from_matrix, euler_from_quaternion

class DartHalfCheetahEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*6,[-1.0]*6])
        self.g_action_scaler = 1.0
        self.action_scale = np.array([120, 90, 60, 120, 60, 30]) * 1.0
        self.train_UP = False
        self.noisy_input = True
        self.randomize_initial_state = True
        obs_dim = 17
        self.tilt_z = 0.0

        self.learn_rear_walk = False   # learn to walk with rear legs only

        self.learn_alternative_walk = False

        self.velrew_weight = 1.0
        self.UP_noise_level = 0.0
        self.resample_MP = True  # whether to resample the model paraeters

        self.actuator_nonlinearity = False
        self.actuator_nonlin_coef = 1.0

        self.param_manager = cheetahParamManager(self)

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)

        self.t = 0

        self.total_dist = []

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history

        dart_env.DartEnv.__init__(self, ['half_cheetah.skel'], 5, obs_dim, self.control_bounds, disableViewer=True, dt=0.01)

        self.default_mass = [bn.mass() for bn in self.robot_skeleton.bodynodes]
        self.default_dampings = [jt.damping_coefficient(0) for jt in self.robot_skeleton.joints]
        self.default_stiffness = [jt.spring_stiffness(0) for jt in self.robot_skeleton.joints]

        self.initial_local_coms = [np.copy(bn.local_com()) for bn in self.robot_skeleton.bodynodes]

        self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_worlds[0].set_collision_detector(3)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 0
        self.act_delay = 0

        #self.param_manager.set_simulator_parameters(self.current_param)

        print('sim parameters: ', self.param_manager.get_simulator_parameters())
        self.current_param = self.param_manager.get_simulator_parameters()
        self.active_param = self.param_manager.activated_param

        # data structure for actuation modeling
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]

        self.add_perturbation = True
        self.perturbation_parameters = [0.01, 60, 2, 50]  # probability, magnitude, bodyid, duration
        self.perturb_offset = [0.0, 0.0, 0.0]

        self.dart_worlds[0].set_collision_detector(3)

        utils.EzPickle.__init__(self)

    def resample_task(self):
        self.param_manager.resample_parameters()
        self.current_param = self.param_manager.get_simulator_parameters()
        self.velrew_weight = np.sign(np.random.randn(1))[0]
        return self.current_param, self.velrew_weight

    def set_task(self, task_params):
        self.param_manager.set_simulator_parameters(task_params[0])
        self.current_param = self.param_manager.get_simulator_parameters()
        self.velrew_weight = task_params[1]

    def pad_action(self, a):
        full_ac = np.zeros(len(self.robot_skeleton.q))
        full_ac[3:] = a
        return full_ac

    def unpad_action(self, a):
        return a[3:]

    def do_simulation(self, tau, n_frames):
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

            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def advance(self, a):
        if self.actuator_nonlinearity:
            a = np.tanh(self.actuator_nonlin_coef * a)
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
        tau[3:] = clamped_control * self.action_scale * self.g_action_scaler
        self.do_simulation(tau, self.frame_skip)

    def about_to_contact(self):
        return False

    def post_advance(self):
        self.dart_world.check_collision()

    def terminated(self):
        s = self.state_vector()

        fall_on_ground = False
        front_foot_on_ground = False
        rear_foot_on_ground = False
        if self.learn_rear_walk:
            permitted_contact_bodies = [self.dart_world.skeletons[0].bodynodes[0], self.robot_skeleton.bodynodes[6]]
        else:
            permitted_contact_bodies = [self.dart_world.skeletons[0].bodynodes[0], self.robot_skeleton.bodynodes[6], self.robot_skeleton.bodynodes[9]]
        contacts = self.dart_world.collision_result.contacts
        for contact in contacts:
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
            if contact.bodynode1 not in permitted_contact_bodies or contact.bodynode2 not in permitted_contact_bodies:
                fall_on_ground = True
            if contact.bodynode1 == self.robot_skeleton.bodynodes[9] or contact.bodynode2 == self.robot_skeleton.bodynodes[9]:
                front_foot_on_ground = True
            if contact.bodynode1 == self.robot_skeleton.bodynodes[6] or contact.bodynode2 == self.robot_skeleton.bodynodes[6]:
                rear_foot_on_ground = True

        if self.learn_alternative_walk:
            if front_foot_on_ground and rear_foot_on_ground:
                return True

        if self.learn_rear_walk or self.learn_alternative_walk:
            done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and np.abs(s[2]) < 1.9)
        else:
            done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and np.abs(s[2]) < 1.3)
        return done or fall_on_ground

    def pre_advance(self):
        self.posbefore = self.robot_skeleton.q[0]

    def reward_func(self, a, step_skip=1):
        posafter = self.robot_skeleton.q[0]
        alive_bonus = 1.0
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight
        reward += alive_bonus * step_skip
        reward -= 1e-1 * np.square(a).sum()

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all())

        if done:
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

        return ob, reward, done, envinfo

    def _get_obs(self, update_buffer = True):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        if self.train_UP:
            UP = self.param_manager.get_simulator_parameters()
            if self.UP_noise_level > 0:
                UP += np.random.uniform(-self.UP_noise_level, self.UP_noise_level, len(UP))
                UP = np.clip(UP, -0.05, 1.05)
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
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]
        qpos = np.array(self.robot_skeleton.q)
        qvel = np.array(self.robot_skeleton.dq)

        if self.randomize_initial_state:
            qpos += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
            qvel += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)

        if self.learn_rear_walk:
            qpos[1] += 0.15
            qpos[2] -= 1.0

        if self.learn_alternative_walk:
            qpos[1] += 0.08
            qpos[2] -= 0.3

        self.set_state(qpos, qvel)
        if self.resample_MP:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()

        self.observation_buffer = []
        self.action_buffer = []

        state = self._get_obs(update_buffer = True)

        self.cur_step = 0

        self.height_threshold_low = 0.56*self.robot_skeleton.bodynodes[2].com()[1]
        self.t = 0

        self.fall_on_ground = False

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4

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
