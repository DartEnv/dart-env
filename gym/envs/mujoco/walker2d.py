import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.dart.parameter_managers import *

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self.include_obs_history = 1
        self.include_act_history = 0

        self.relaxed_bound = True

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

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.state_buffer = []

        self.obs_delay = 2
        self.act_delay = 0

        self.cur_step = 0
        self.use_sparse_reward = False
        self.horizon = 999
        self.total_reward = 0

        self.param_manager = mjWalkerParamManager(self)

        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)

        # self.param_manager.set_simulator_parameters([0.7, 0.4])

        utils.EzPickle.__init__(self)

    def pre_advance(self):
        self.posbefore = self.sim.data.qpos[0]

    def advance(self, a):
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay - 1]

        self.do_simulation(a, self.frame_skip)

    def reward_func(self, a):
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - self.posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        if self.use_sparse_reward:
            self.total_reward += reward
            reward = 0.0
            if self.terminated():
                reward = self.total_reward

        return reward

    def post_advance(self):
        pass

    def terminated(self):
        posafter, height, ang = self.sim.data.qpos[0:3]
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        if self.relaxed_bound:
            done = not (height > 0.3 and height < 2.0 and
                        ang > -1.6 and ang < 1.6)
        if self.cur_step >= self.horizon:
            done = True

        return done

    def step(self, a):
        self.cur_step += 1
        self.pre_advance()
        self.advance(a)
        reward = self.reward_func(a)

        # uncomment to enable knee joint limit penalty
        joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.model.jnt_range[j][0] - self.sim.data.qpos[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.model.jnt_range[j][1] - self.sim.data.qpos[j]) < 0.05:
                joint_limit_penalty += abs(1.5)
        reward -= 5e-1*joint_limit_penalty

        done = self.terminated()
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self, update_buffer=True):
        state = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

        if update_buffer:
            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0]])

        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[-1 - i]])
            else:
                final_obs = np.concatenate([final_obs, [0.0] * 6])

        return final_obs

    def reset_model(self):
        qpos = np.array(self.init_qpos)# + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = np.array(self.init_qvel)# + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        if np.random.random() > 0.5:
            qpos[3] += 0.3
            qpos[6] -= 0.3
        else:
            qpos[3] -= 0.3
            qpos[6] += 0.3

        # qvel[0] = 2.5

        self.set_state(
            qpos,
            qvel
        )

        self.cur_step = 0
        self.total_reward = 0

        self.observation_buffer = []
        self.action_buffer = []

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
