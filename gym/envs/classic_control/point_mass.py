import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import copy

logger = logging.getLogger(__name__)

class PointMassEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low = np.array([-1, -1]), high = np.array([1, 1]))
        self.dt = 0.3

        self.randomize_initial_state = False

        self.mass = 5.0

        self.alive_bonus = 0.0

        self.observation_space = spaces.Box(np.array([-10, -10, -np.inf, -np.inf]), np.array([10, 10, np.inf, np.inf]))

        self.input_time = False

        if self.input_time:
            self.observation_space = spaces.Box(np.array([-10, -10, -np.inf, -np.inf, 0.0]),
                                                np.array([10, 10, np.inf, np.inf, np.inf]))

        self.init_pose = np.array([-2.0, -2.0, 0.0, 0.0])
        self.targets = [np.array([0.0, 10.0])]

        self.seed()
        self.viewer = None
        self.state = None

        self.targets = [np.array([-5.0, 0.0])]

        self.perturb_zones = [np.array([-5.0, 0.0, 1.0]), np.array([0.0, 8.0, 2.0]), np.array([8.0, -3.0, 1.5]),
                           np.array([-3.0, -3.0, 0.5]), np.array([-6.0, -4.0, 1.2]), np.array([-8.0, 3.0, 1.7]),
                           np.array([-5.0, -9.0, 1.3]), np.array([1.0, -0.0, 1.5]), np.array([3.0, -9.0, 1.5])]

        self.prediction_states = []

        self.paths = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def computer_rew(self):
        reward = -0.05 - 3 * np.linalg.norm(self.state[0:2] - self.targets[0])
        if np.linalg.norm(self.state[0:2] - self.targets[0]) < 1.0:
            reward += 25 - 10 * np.linalg.norm(self.state[0:2] - self.targets[0])
        return reward

    def step(self, action):
        action = np.clip(action, -1, 1)

        rew_before = self.computer_rew()

        state_before = np.copy(self.state)

        perturb = np.zeros(2)
        for perturb_zone in self.perturb_zones:
            if np.linalg.norm(self.state[0:2] - perturb_zone[0:2]) < perturb_zone[2]:
                perturb = self.state[0:2] - perturb_zone[0:2]
                break

        self.state[2:] += action * self.dt / self.mass + perturb * 0.2
        self.state[2:] = np.clip(self.state[2:], -0.5, 0.5)
        self.state[0:2] += self.state[2:] * self.dt
        self.state = self.state.clip(-10, 10)

        rew_after = self.computer_rew()

        reward = rew_after - rew_before + self.alive_bonus

        # MMHACK
        reward = 0.0

        done = False

        self.current_action = np.copy(action)

        obs = self._get_obs()

        self.vec = self.targets[0] - self.state[0:2]

        self.cur_step += 1

        self.paths.append([state_before, self.state, self.current_mode])

        return obs, reward, done, {}

    def _get_obs(self):
        obs = np.array(self.state)# + np.random.uniform(-0.2, 0.2, len(self.state))
        if self.input_time:
            obs = np.concatenate([obs, [self.dt * self.cur_step]])
        return obs


    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        screen_width = 500
        screen_height = 500

        world_width = 20.0
        scale = screen_width/world_width
        offset = np.array([screen_height / 2.0, screen_width / 2.0])

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)


            agent = rendering.make_circle(radius=0.5*scale)
            self.agent_transform = rendering.Transform(self.state[0:2] * scale + offset)
            agent.add_attr(self.agent_transform)
            agent.set_color(0.25, 0.25, 1.0)
            self.viewer.add_geom(agent)

            self.target_transforms = []
            for target_id in range(len(self.targets)):
                target1 = rendering.make_circle(radius=0.5*scale)
                self.target_transforms.append(rendering.Transform(self.targets[target_id] * scale + offset))
                target1.add_attr(self.target_transforms[-1])
                target1.set_color(1.0, 0.8, 0.25)
                self.viewer.add_geom(target1)

            for plot_id in range(len(self.perturb_zones)):
                plot_obj = rendering.make_circle(radius=self.perturb_zones[plot_id][2]*scale)
                transform = rendering.Transform(self.perturb_zones[plot_id][0:2] * scale + offset)
                plot_obj.add_attr(transform)
                plot_obj.set_color(0.2, 0.5, 0.5)
                self.viewer.add_geom(plot_obj)


            self.agent = agent

        if self.state is None: return None

        new_pos = self.state[0:2] * scale + offset
        self.agent_transform.set_translation(new_pos[0], new_pos[1])

        for target_id in range(len(self.targets)):
            new_target = self.targets[target_id] * scale + offset
            self.target_transforms[target_id].set_translation(new_target[0], new_target[1])

        if len(self.prediction_states) > 0:
            for i in range(0, len(self.prediction_states)-1):
                start = np.array(self.prediction_states[i][0:2]) * scale + offset
                end = np.array(self.prediction_states[i+1][0:2]) * scale + offset
                line = self.viewer.draw_line(start, end)
                line.attrs[1].stroke = 15
                line.attrs[0].vec4 = (0.0, 1.0, 0.0, 0.5)

        if len(self.paths) > 2:
            for i in range(1, len(self.paths)):
                start = self.paths[i][0][0:2] * scale + offset
                end = self.paths[i][1][0:2] * scale + offset
                line = self.viewer.draw_line(start, end)
                line.attrs[1].stroke = 15
                line.attrs[0].vec4 = (0.0, 0.0, 1.0, 0.2)


        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) * 0 + 0.0001

        self.state += self.init_pose

        self.current_action = np.ones(2)

        self.vec = self.targets[0] - self.state[0:2]

        self.cur_step = 0

        self.current_mode = 'NORMAL'

        # self.paths = []

        return self._get_obs()

    def state_vector(self):
        return np.copy(self.state)

    def set_state_vector(self, s):
        self.state = np.copy(s)

    # set state from observation
    def set_from_obs(self, obs):
        self.set_state_vector(obs)