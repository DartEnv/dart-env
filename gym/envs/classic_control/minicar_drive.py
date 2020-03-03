import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class MiniCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low = np.array([-1, -1]), high = np.array([1, 1]))
        self.dt = 0.3

        self.randomize_initial_state = True

        self.mass = 10.0
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

        self.init_pose = np.zeros(4)
        self.init_pose[2] = 0.1
        self.targets = [np.array([9.0, 0.0])]

        if self.input_target:
            self.observation_space = spaces.Box(np.array([-10, -10, -np.inf, -np.inf, -10, -10]), np.array([10, 10, np.inf, np.inf, 10, 10]))

        self._seed()
        self.viewer = None
        self.state = None

        self.safe_zones = []
        self.warn_zones = []

        # self.safe_zones = [np.array([[-7, -1.5], [-7, 1.5], [7, 1.5], [7, -1.5]]),
        #                    np.array([[6.5, -5.5], [6.5, 5.5], [8, 5.5], [8, -5.5]])]
        # self.warn_zones = [np.array([[-9.5, -3], [-9.5, 3], [9.5, 3], [9.5, -3]]),
        #                    np.array([[4.5, -7.5], [4.5, 7.5], [10.0, 7.5], [10.0, -7.5]])]

        # maze v1
        # self.safe_zones = [np.array([[-10.5, -10.5], [-10.5, 10.5], [10.5, 10.5], [10.5, -10.5]])]
        # self.warn_zones = [np.array([[3.5, -6.5], [3.5, 6.5], [8.5, 6.5], [8.5, -6.5]])]
        # self.forbidden_zones = [np.array([[5, -5.0], [5, 5.0], [7, 5.0], [7, -5.0]])]
        # self.zone_orders = ['FORBIDDEN', 'WARN', 'SAFE']  # from top to bottom

        # maze v2
        self.init_pose = np.array([-7.0, -6.5, 0, 0])
        self.targets = [np.array([-7.0, 6])]
        self.safe_zones = [np.array([[-7.5, -7.5], [-7.5, -5.5], [7.5, -5.5], [7.5, -7.5]]),
                           np.array([[7.5, -7.5], [7.5, 7.5], [5.5, 7.5], [5.5, -7.5]]),
                           np.array([[-7.5, 7.5], [-7.5, 5.5], [7.5, 5.5], [7.5, 7.5]])]
        self.warn_zones = [np.array([[-9, -9], [-9, -4], [9, -4], [9, -9]]),
                           np.array([[9, -9], [9, 9], [4, 9], [4, -9]]),
                           np.array([[-9, 9], [-9, 4], [9, 4], [9, 9]])]
        self.forbidden_zones = [np.array([[-10.5, -10.5], [-10.5, 10.5], [10.5, 10.5], [10.5, -10.5]])]
        self.zone_orders = ['SAFE', 'WARN', 'FORBIDDEN']  # from top to bottom

    # whether the agent is in safe zone
    def in_safe_zones(self):
        for zone in self.safe_zones:
            if self.state[0] > np.min(zone[:, 0]) and self.state[0] < np.max(zone[:, 0]) \
                    and self.state[1] > np.min(zone[:, 1]) and self.state[1] < np.max(zone[:, 1]):
                return True
        return False


    # whether the agent is in warn zone
    def in_warn_zones(self):
        for zone in self.warn_zones:
            if self.state[0] > np.min(zone[:, 0]) and self.state[0] < np.max(zone[:, 0]) \
                    and self.state[1] > np.min(zone[:, 1]) and self.state[1] < np.max(zone[:, 1]):
                return True
        return False

    # whether the agent is in forbidden zone
    def in_forbidden_zones(self):
        for zone in self.forbidden_zones:
            if self.state[0] > np.min(zone[:, 0]) and self.state[0] < np.max(zone[:, 0]) \
                    and self.state[1] > np.min(zone[:, 1]) and self.state[1] < np.max(zone[:, 1]):
                return True
        return False


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

    def computer_rew(self):
        reward = -0.05 - 3 * np.linalg.norm(self.state[0:2] - self.targets[0])
        if np.linalg.norm(self.state[0:2] - self.targets[0]) < 0.8:
            reward += 25 - 3 * np.linalg.norm(self.state[0:2] - self.targets[0])
        return reward

    def step(self, action):
        action = action.clip(-1, 1)

        if self.in_warn_zones():
            action *= 1.0

        rew_before = self.computer_rew()

        self.state[2:] += action * self.dt / self.mass + self.wind

        cur_vel = np.linalg.norm(self.state[2:])
        cur_vel += action[0] * self.dt / self.mass  # put gas or break
        cur_vel = np.clip(cur_vel, -0.7, 0.7)

        cur_ori = self.state[2:] / np.linalg.norm(self.state[2:])
        rot = action[1] * 0.6
        cur_ori = np.array([np.cos(rot) * cur_ori[0] - np.sin(rot) * cur_ori[1], np.sin(rot) * cur_ori[0] + np.cos(rot) * cur_ori[1]])
        self.state[2:] = cur_ori * cur_vel

        self.state[0:2] += self.state[2:] * self.dt
        self.state = self.state.clip(-10, 10)

        rew_after = self.computer_rew()

        reward = rew_after - rew_before# - np.sum(np.abs(action)) * 1.0

        done = False

        state = 0
        for zone in self.zone_orders:
            if zone == 'SAFE' and self.in_safe_zones():
                state = 1
                break
            if zone == 'WARN' and self.in_warn_zones():
                state = 2
                break
            if zone == 'FORBIDDEN' and self.in_forbidden_zones():
                state = 3
                break

        if state == 3:
            done = True

        self.current_action = np.copy(action)

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
        self.state += self.init_pose

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

    def _draw_zones(self, zones, scale, offset, r, g, b):
        from gym.envs.classic_control import rendering
        for i in range(len(zones)):
            zone = rendering.make_polygon(zones[i] * scale + offset)
            zone.set_color(r, g, b)
            self.viewer.add_geom(zone)

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

            for z in range(len(self.zone_orders)-1, -1, -1):
                if self.zone_orders[z] == 'WARN':
                    self._draw_zones(self.warn_zones, scale, offset, 0.85, 1.0, 0.85)
                elif self.zone_orders[z] == 'SAFE':
                    self._draw_zones(self.safe_zones, scale, offset, 0.85, 1.0, 1.0)
                elif self.zone_orders[z] == 'FORBIDDEN':
                    self._draw_zones(self.forbidden_zones, scale, offset, 0.3, 0.3, 0.3)

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

            self.agent = agent

        if self.state is None: return None

        new_pos = self.state[0:2] * scale + offset
        self.agent_transform.set_translation(new_pos[0], new_pos[1])

        new_target = self.targets[0] * scale + offset
        self.target_transform.set_translation(new_target[0], new_target[1])

        line = self.viewer.draw_line(new_pos, new_pos + np.array(self.state[2:]) * scale)
        line.set_color(1, 0, 0)
        self.viewer.add_onetime(line)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def state_vector(self):
        return np.copy(self.state)

    def set_state_vector(self, s):
        self.state = np.copy(s)