import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.dart.parameter_managers import *

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.train_UP = False
        self.noisy_input = False

        self.resample_MP = False  # whether to resample the model paraeters
        self.param_manager = mjcheetahParamManager(self)
        self.velrew_weight = 1.0

        self.include_obs_history = 1
        self.include_act_history = 0

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 0
        self.act_delay = 0
        self.tilt_z = 0

        self.current_step = 0
        self.max_step = 1000

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def resample_task(self):
        self.velrew_weight = np.sign(np.random.randn(1))[0]

    def pad_action(self, a):
        full_ac = np.zeros(len(self.sim.data.qpos))
        full_ac[3:] = a
        return full_ac

    def unpad_action(self, a):
        return a[3:]

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
        if self.current_step > self.max_step:
            return True
        return False

    def pre_advance(self):
        self.posbefore = self.sim.data.qpos[0]

    def reward_func(self, a):
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight
        reward += alive_bonus
        reward -= 0.1 * np.square(a).sum()
        return reward


    def step(self, a):
        self.pre_advance()
        self.advance(a)
        self.post_advance()

        reward = self.reward_func(a)

        done = self.terminated()

        self.current_step += 1

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self, update_buffer = True):
        state = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])
        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])
        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        if update_buffer:
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
                final_obs = np.concatenate([final_obs, [0.0] * 6])

        return final_obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        self.observation_buffer = []
        self.action_buffer = []

        self.current_step = 0

        if self.resample_MP:
            self.param_manager.resample_parameters()

        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
