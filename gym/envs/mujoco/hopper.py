import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.dart.parameter_managers import *

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.train_UP = False
        self.noisy_input = False

        self.relaxed_bound = True

        self.resample_MP = False  # whether to resample the model paraeters
        self.param_manager = mjHopperManager(self)
        self.velrew_weight = 1
        self.velrew_input = False
        self.two_pose_input = False
        self.previous_pose = np.zeros(5)
        self.input_time = False

        self.test_mode = True

        # self.obs_ranges = np.array([[0.6, 1.8], [-0.5, 0.5], ])

        self.randomize_history_input = False
        self.history_buffers = []

        self.terminate_for_not_moving = None#[0.5, 1.0]

        self.pid_controller = None#[250, 25]
        if self.pid_controller is not None:
            self.torque_limit = [-200, 200]
            self.action_scale = np.array([np.pi / 3.0, np.pi / 3.0, np.pi / 3.0])

        self.include_obs_history = 1
        self.include_act_history = 0

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 1
        self.act_delay = 0

        self.cur_step = 0

        self.use_sparse_reward = False
        self.horizon = 999

        self.total_reward = 0

        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)

        self.param_manager.set_simulator_parameters([0.1])

        utils.EzPickle.__init__(self)

    def pad_action(self, a):
        full_ac = np.zeros(len(self.sim.data.qpos))
        full_ac[3:] = a
        return full_ac

    def unpad_action(self, a):
        return a[3:]

    def do_simulation(self, ctrl, n_frames):
        if self.pid_controller is not None:
            target_angles = np.copy(ctrl)

        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            if self.pid_controller is not None:
                jpos = np.array(self.sim.data.qpos)[3:]
                jvel = np.array(self.sim.data.qvel)[3:]

                torque = self.pid_controller[0] * (target_angles - jpos) - self.pid_controller[1] * jvel
                clipped_torque = np.clip(torque, self.torque_limit[0], self.torque_limit[1])

                self.sim.data.ctrl[:] = clipped_torque
            self.sim.step()

    def advance(self, a):
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay - 1]

        self.do_simulation(a, self.frame_skip)

    def about_to_contact(self):
        return False

    def post_advance(self):
        pass

    def terminated(self):
        s = self.state_vector()
        height, ang = self.sim.data.qpos[1:3]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .8))
        if self.relaxed_bound:
            done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                        (height > .4) and (abs(ang) < 1.0))


        if self.cur_step >= self.horizon:
            done = True

        if self.terminate_for_not_moving is not None:
            if self.cur_step * self.dt > self.terminate_for_not_moving[1] and \
                    (np.abs(s[0]) < self.terminate_for_not_moving[0] or
                     s[0] * self.velrew_weight < 0):
                done = True


        return done

    def pre_advance(self):
        self.posbefore = self.sim.data.qpos[0]

    def reward_func(self, a):
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        joint_limit_penalty = 0
        if np.abs(self.sim.data.qpos[-2]) < 0.05:
            joint_limit_penalty += 1.5
        # reward -= 5e-1 * joint_limit_penalty
        if self.use_sparse_reward:
            self.total_reward += reward
            reward = 0.0
            if self.terminated():
                reward = self.total_reward

        return reward

    def step(self, a):
        # a = -np.copy(a)
        self.cur_step += 1
        self.pre_advance()
        self.advance(a)
        self.post_advance()
        # MMHACK
        # if hasattr(self, "env_change_parameters"):
        #     if self.cur_step > self.env_change_parameters[0] and self.cur_step < self.env_change_parameters[0] + self.env_change_parameters[1]:
        #         self.model.actuator_gear[0][0] = 150
        #         self.model.actuator_gear[1][0] = 150
        #         self.model.actuator_gear[2][0] = 150
        #     else:
        #         self.model.actuator_gear[0][0] = 300
        #         self.model.actuator_gear[1][0] = 300
        #         self.model.actuator_gear[2][0] = 300

        reward = self.reward_func(a)

        done = self.terminated()
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self, update_buffer = True):
        state = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

        state[1] = (state[1] + np.pi) % (2 * np.pi) - np.pi

        if self.two_pose_input:
            state = np.concatenate([
                self.sim.data.qpos.flat[1:],
                (self.previous_pose - self.sim.data.qpos.flat[1:]) / self.dt
            ])
            self.previous_pose = self.sim.data.qpos.flat[1:]

        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])

        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        if update_buffer:
            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        current_obs = np.array([])  # the current part of observation
        history_obs = np.array([])  # the history part of the observation
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
                if i == 0:
                    current_obs = np.concatenate([history_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
                if i > 0 and self.randomize_history_input:
                    history_obs = np.concatenate([history_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0]])
                if i == 0:
                    current_obs = np.concatenate([history_obs, self.observation_buffer[0]])
                if i > 0 and self.randomize_history_input:
                    history_obs = np.concatenate([history_obs, self.observation_buffer[0]])
        if self.randomize_history_input:
            self.history_buffers.append(history_obs)
            final_obs = np.concatenate(
                [current_obs, self.history_buffers[np.random.randint(len(self.history_buffers))]])

        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[-1 - i]])
            else:
                final_obs = np.concatenate([final_obs, [0.0] * 3])

        if self.velrew_input:
            final_obs = np.concatenate([final_obs, [self.velrew_weight]])

        if self.input_time:
            final_obs = np.concatenate([final_obs, [self.cur_step * self.dt]])

        return final_obs

    def get_lowdim_obs(self):
        full_obs = self._get_obs(update_buffer=False)
        return np.array([full_obs[5], full_obs[6]])

    def reset_model(self):
        qpos = np.array(self.init_qpos)# + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = np.array(self.init_qvel)# + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        if not self.test_mode:
            qpos += self.np_random.uniform(low=-.05, high=.05, size=self.model.nq)
            qvel += self.np_random.uniform(low=-.05, high=.05, size=self.model.nv)
        # qvel[0] += 2.0
        # qpos[2] = 0.1
        # print(qpos)
        # qpos[1] -= 0.5
        # qpos[2] = 1.0
        # qpos[5] = 0.5

        self.set_state(qpos, qvel)

        self.observation_buffer = []
        self.action_buffer = []

        self.cur_step = 0
        self.total_reward = 0

        self.history_buffers = []

        if self.resample_MP:
            self.param_manager.resample_parameters()

        self.previous_pose = np.array(self.sim.data.qpos.flat[1:])

        self.env_change_parameters = [150, 350]#[np.random.randint(150, 500), np.random.randint(300, 400)] # start frame, length

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
