import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class PointReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low = np.array([-1, -1]), high = np.array([1, 1]))
        self.dt = 0.3

        self.randomize_initial_state = True

        self.forbidden_zones = []

        self.mass = 0.7
        self.mass_range = [0.2, 10.0]

        self.wind = np.array([0.0, 0.0])

        self.input_target = False
        self.train_UP = False
        self.resample_MP = False

        self.resample_task_on_reset = False

        if not self.train_UP:
            self.observation_space = spaces.Box(np.array([-10, -10, -np.inf, -np.inf]), np.array([10, 10, np.inf, np.inf]))
        else:
            self.observation_space = spaces.Box(np.concatenate([[-10, -10, -np.inf, -np.inf], [-np.inf] * 1]),
                                                np.concatenate([[10, 10, np.inf, np.inf], [np.inf] * 1]))

        self.targets = [np.array([5.0, 4.0])]

        if self.input_target:
            self.observation_space = spaces.Box(np.array([-10, -10, -np.inf, -np.inf, -10, -10]), np.array([10, 10, np.inf, np.inf, 10, 10]))

        self._seed()
        self.viewer = None
        self.state = None

    def resample_task(self):
        # A point on half circle above
        radius = np.random.uniform(6.8, 7.0)
        # angle = np.random.uniform(0.0, np.pi)
        angle = np.random.choice(np.arange(0, np.pi/2.0+0.01, np.pi / 10.0))
        # if np.random.random() < 0.05:
        #     angle = - np.pi/2.0
        self.targets = [np.array([np.sin(angle) * radius, np.cos(angle) * radius])]

        # MMHACK: vary target and mass
        # rand = np.random.random()
        # if rand < 0.25:
        #     self.targets = [np.array([5.0, 4.0])]
        #     self.mass = 0.3
        # elif rand < 0.5:
        #     self.targets = [np.array([-5.0, 3.0])]
        #     self.mass = 1.0
        # elif rand < 0.75:
        #     self.targets = [np.array([6.0, -3.0])]
        #     self.mass = 1.7
        # else:
        #     self.targets = [np.array([-4.0, -5.0])]
        #     self.mass = 2.4
        # MMHACK: vary forbidden zone and target
        # if rand < 0.25:
        #     self.targets = [np.array([7.0, 0.0])]
        #     self.forbidden_zones = [[np.array([-2, -4]), np.array([-1, 4])]]
        # elif rand < 0.5:
        #     self.targets = [np.array([-6.0, 0.0])]
        #     self.forbidden_zones = [[np.array([2, -4]), np.array([1, 4])]]
        # elif rand < 0.75:
        #     self.targets = [np.array([0.0, 6.0])]
        #     self.forbidden_zones = [[np.array([-4, -2]), np.array([4, -1])]]
        # else:
        #     self.targets = [np.array([0.0, -5.0])]
        #     self.forbidden_zones = [[np.array([4, 2]), np.array([-4, 1])]]

        return [np.copy(self.targets), self.mass, np.copy(self.forbidden_zones)]

    def set_task(self, task_params):
        self.targets = np.copy(task_params[0])
        self.mass = task_params[1]
        self.forbidden_zones = task_params[2]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action.clip(-1, 1)

        self.state[2:] += action * self.dt / self.mass + self.wind
        self.state[0:2] += self.state[2:] * self.dt
        self.state = self.state.clip(-10, 10)

        reward = -0.05 - 3 * np.linalg.norm(self.state[0:2] - self.targets[0]) - np.sum(np.abs(action)) * 5.0
        if np.linalg.norm(self.state[0:2] - self.targets[0]) < 0.8:
            reward += 25 - 3 * np.linalg.norm(self.state[0:2] - self.targets[0])

        done = False

        self.current_action = np.copy(action)

        if len(self.forbidden_zones) > 0:
            for zone in self.forbidden_zones:
                if (self.state[0] - zone[0][0]) * (self.state[0] - zone[1][0]) <= 0.0 and \
                    (self.state[1] - zone[0][1]) * (self.state[1] - zone[1][1]) <= 0.0:
                    self.state[0:2] = -self.targets[0]

        obs = np.array(self.state)

        if self.train_UP:
            obs = np.concatenate([obs, [self.mass]*1])

        if self.input_target:
            obs = np.concatenate([obs, self.targets[0] / 10.0])

        return obs, reward, done, {}

    def reset(self):
        if not self.randomize_initial_state:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) * 0 + 0.0001
        else:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
            self.state[0] = self.np_random.uniform(-0.5, 0.5)
            self.state[1] = self.np_random.uniform(-0.5, 0.5)


        self.current_action = np.ones(2)

        if self.resample_MP:
            self.mass = np.random.uniform(self.mass_range[0], self.mass_range[1])

        if self.resample_task_on_reset:
            self.resample_task()

        if self.train_UP:
            obs = np.concatenate([self.state, [self.mass]*1])
            # obs = np.concatenate([self.state, [1.5]])
        else:
            obs = np.copy(self.state)

        if self.input_target:
            obs = np.concatenate([obs, self.targets[0] / 10.0])

        return obs

    def _get_obs(self):
        return np.copy(self.state)

    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 500
        screen_height = 500

        world_width = 20.0
        scale = screen_width/world_width
        offset = np.array([screen_height / 2.0, screen_width / 2.0])

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(radius=0.2*scale)
            self.agent_transform = rendering.Transform(self.state[0:2] * scale + offset)
            agent.add_attr(self.agent_transform)
            agent.set_color(0.5, 0.5, 1.0)
            self.viewer.add_geom(agent)

            target1 = rendering.make_circle(radius=0.2*scale)
            self.target_transform = rendering.Transform(self.targets[0] * scale + offset)
            target1.add_attr(self.target_transform)
            target1.set_color(0.5, 1.0, 0.5)
            self.viewer.add_geom(target1)

        for zone in self.forbidden_zones:
            v1 = zone[0] * scale + offset
            v3 = zone[1] * scale + offset
            v2 = np.array([v1[0], v3[1]])
            v4 = np.array([v3[0], v1[1]])
            zone_poly = rendering.make_polygon([v1, v2, v3, v4])
            zone_poly.set_color(0.3, 0.3, 0.3)
            self.viewer.add_onetime(zone_poly)

        if self.state is None: return None

        new_pos = self.state[0:2] * scale + offset
        self.agent_transform.set_translation(new_pos[0], new_pos[1])

        new_target = self.targets[0] * scale + offset
        self.target_transform.set_translation(new_target[0], new_target[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def state_vector(self):
        return np.copy(self.state)

    def set_state_vector(self, s):
        self.state = np.copy(s)