import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import copy

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

        self.mass = 5.0
        self.mass_range = [3.0, 8.0]

        self.wind = np.array([0.0, 0.0])

        self.input_target = True
        self.train_UP = False
        self.resample_MP = False

        self.test_mode = False

        self.alive_bonus = 0.0

        self.draw_perception = True

        self.draw_path = False

        self.resample_task_on_reset = False

        self.current_mode = 'NORMAL'

        if not self.train_UP:
            self.observation_space = spaces.Box(np.array([-10, -10, -np.inf, -np.inf]), np.array([10, 10, np.inf, np.inf]))
        else:
            self.observation_space = spaces.Box(np.concatenate([[-10, -10, -np.inf, -np.inf], [-np.inf] * 1]),
                                                np.concatenate([[10, 10, np.inf, np.inf], [np.inf] * 1]))

        self.init_pose = np.array([-8.0, -8.0, 0.0, 0.0])
        self.targets = [np.array([0.0, 10.0])]

        if self.input_target:
            self.observation_space = spaces.Box(np.array([-np.inf, -np.inf, -10, -10, -10, -10, -10, -10]),
                                                np.array([np.inf, np.inf, 10, 10, 10, 10, 10, 10]))

        self._seed()
        self.viewer = None
        self.state = None

        self.safe_zones = []
        self.warn_zones = []

        # self.safe_zones = [np.array([[-7, -1.5], [-7, 1.5], [7, 1.5], [7, -1.5]]),
        #                    np.array([[6.5, -5.5], [6.5, 5.5], [8, 5.5], [8, -5.5]])]
        # self.warn_zones = [np.array([[-9.5, -3], [-9.5, 3], [9.5, 3], [9.5, -3]]),
        #                    np.array([[4.5, -7.5], [4.5, 7.5], [10.0, 7.5], [10.0, -7.5]])]

        # no maze
        self.safe_zones = []
        self.warn_zones = []
        # self.forbidden_zones = []
        self.zone_orders = ['FORBIDDEN', 'WARN', 'SAFE']
        self.safe_zones = [np.array([[-10.5, -10.5], [-10.5, 10.5], [10.5, 10.5], [10.5, -10.5]])]
        self.warn_zones = [np.array([[-10.5, -10.5], [-10.5, 10.5], [10.5, 10.5], [10.5, -10.5]])]
        self.forbidden_zones = [np.array([[-10, -10.0], [-10, 10.0], [-9.5, 10.0], [-9.5, -10.0]]),
                                np.array([[-10, -10.0], [-10, -9.5], [10, -9.5], [10, -10.0]]),
                                np.array([[-10, 10.0], [-10, 9.5], [10, 9.5], [10, 10.0]]),
                                np.array([[9.5, -10.0], [9.5, 10], [10, 10], [10, -10.0]])
                                ]
        self.forbidden_spheres = [[np.array([-5.0, -4.0]), 1.0], [np.array([10.0, 8.0]), 0.8],
                                  [np.array([2.0, 4.5]), 1.5], [np.array([2.0, -2.0]), 2.0]]
        self.targets = [np.array([8.0, 8.0])]

        # perturb obstacle positions
        # np.random.seed(15)
        # for sp in self.forbidden_spheres:
        #     sp[0] += np.random.uniform(-1.5, 1.5, 2)
        # self.forbidden_spheres[2][0][0] -= 1.5
        # # self.forbidden_spheres[2][0][1] += 0.5
        # self.forbidden_spheres[3][0][0] -= 0.05
        # self.forbidden_spheres[3][0][1] -= 0.0

        self.forbidden_spheres_original = copy.deepcopy(self.forbidden_spheres)

        # maze v1
        # self.safe_zones = [np.array([[-10.5, -10.5], [-10.5, 10.5], [10.5, 10.5], [10.5, -10.5]])]
        # self.warn_zones = [np.array([[3.5, -6.5], [3.5, 6.5], [8.5, 6.5], [8.5, -6.5]])]
        # self.forbidden_zones = [np.array([[5, -5.0], [5, 5.0], [7, 5.0], [7, -5.0]])]
        # self.zone_orders = ['FORBIDDEN', 'WARN', 'SAFE']  # from top to bottom

        # maze v2
        # self.init_pose = np.array([-7.0, -6.5, 0, 0])
        # self.targets = [np.array([-7.0, 6])
        # self.safe_zones = [np.array([[-7.5, -7.5], [-7.5, -5.5], [7.5, -5.5], [7.5, -7.5]]),
        #                    np.array([[7.5, -7.5], [7.5, 7.5], [5.5, 7.5], [5.5, -7.5]]),
        #                    np.array([[-7.5, 7.5], [-7.5, 5.5], [7.5, 5.5], [7.5, 7.5]])]
        # self.warn_zones = [np.array([[-9, -9], [-9, -4], [9, -4], [9, -9]]),
        #                    np.array([[9, -9], [9, 9], [4, 9], [4, -9]]),
        #                    np.array([[-9, 9], [-9, 4], [9, 4], [9, 9]])]
        # self.forbidden_zones = [np.array([[-10.5, -10.5], [-10.5, 10.5], [10.5, 10.5], [10.5, -10.5]])]
        # self.zone_orders = ['SAFE', 'WARN', 'FORBIDDEN']  # from top to bottom

    # whether inside forbidden sphere
    # return binary value and nearest point if not in collision
    def in_forbidden_spheres(self):
        closest_vec = np.array([30, 30.0, 30.0, 30.0])
        for sph in self.forbidden_spheres:
            dist = np.linalg.norm(self.state[0:2] - sph[0])
            if dist < sph[1]:
                return True, np.array([0, 0.0, 0.0, 0.0])
            else:
                if np.linalg.norm(closest_vec[0:2]) > dist - sph[1]:
                    dir = sph[0] - self.state[0:2]
                    dir /= np.linalg.norm(dir)
                    closest_vec[2:] = closest_vec[0:2]
                    closest_vec[0:2] = dir * (dist - sph[1])
                elif np.linalg.norm(closest_vec[2:]) > dist - sph[1]:
                    dir = sph[0] - self.state[0:2]
                    dir /= np.linalg.norm(dir)
                    closest_vec[2:] = dir * (dist - sph[1])
        return False, closest_vec


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
        # radius = np.random.uniform(6.8, 7.0)
        # # angle = np.random.uniform(0.0, np.pi)
        # angle = np.random.choice(np.arange(0, np.pi/2.0+0.01, np.pi / 10.0))
        # # if np.random.random() < 0.05:
        # #     angle = - np.pi/2.0
        # self.targets = [np.array([np.sin(angle) * radius, np.cos(angle) * radius])]

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

        self.forbidden_spheres = copy.deepcopy(self.forbidden_spheres_original)
        for sp in self.forbidden_spheres:
            sp[0] += np.random.uniform(-1.5, 1.5, 2)

        return [copy.deepcopy(self.forbidden_spheres)]

    def set_task(self, task_params):
        self.forbidden_spheres = copy.deepcopy(task_params[0])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def computer_rew(self):
        reward = -0.05 - 3 * np.linalg.norm(self.state[0:2] - self.targets[0])
        if np.linalg.norm(self.state[0:2] - self.targets[0]) < 1.0:
            reward += 25 - 10 * np.linalg.norm(self.state[0:2] - self.targets[0])
        return reward

    def step(self, action):
        action = action.clip(-1, 1)

        if self.in_warn_zones():
            action *= 1.0

        rew_before = self.computer_rew()

        if not self.test_mode:
            self.wind = np.random.uniform(-0.1, 0.1, 2)

        state_before = np.copy(self.state)

        self.state[2:] += action * self.dt / self.mass + self.wind
        self.state[2:] = np.clip(self.state[2:], -0.5, 0.5)
        self.state[0:2] += self.state[2:] * self.dt
        self.state = self.state.clip(-10, 10)

        rew_after = self.computer_rew()

        reward = rew_after - rew_before + self.alive_bonus# - np.sum(np.abs(action)) * 1.0

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

        obs = np.array(self.state[2:])

        if self.train_UP:
            obs = np.concatenate([obs, [self.mass]*1])

        if self.input_target:
            obs = np.concatenate([obs, self.targets[0] - self.state[0:2]])

        sphere_collide, closest_vec = self.in_forbidden_spheres()

        if sphere_collide:
            done = True
        if self.input_target:
            obs = np.concatenate([obs, closest_vec])

        self.vec = self.targets[0] - self.state[0:2]

        self.closest_vec = closest_vec

        self.cur_step += 1

        self.paths.append([state_before, self.state, self.current_mode])

        # for sp in self.forbidden_spheres:
        #     sp[0] += np.array([-(self.cur_step * 1.0 % 40) / 39 + 0.5, (self.cur_step * 1.0 % 40) / 39 - 0.5]) * 0.4

        return obs, reward, done, {}

    def _get_obs(self):
        if self.train_UP:
            obs = np.concatenate([self.state, [self.mass] * 1])
        else:
            obs = np.copy(self.state[2:])

        if self.input_target:
            obs = np.concatenate([obs, (self.targets[0] - self.state[0:2])])

        sphere_collide, closest_vec = self.in_forbidden_spheres()
        if self.input_target:
            obs = np.concatenate([obs, closest_vec])
        return obs

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

            agent = rendering.make_circle(radius=0.5*scale)
            self.agent_transform = rendering.Transform(self.state[0:2] * scale + offset)
            agent.add_attr(self.agent_transform)
            agent.set_color(0.25, 0.25, 1.0)
            self.viewer.add_geom(agent)

            target1 = rendering.make_circle(radius=0.5*scale)
            self.target_transform = rendering.Transform(self.targets[0] * scale + offset)
            target1.add_attr(self.target_transform)
            target1.set_color(1.0, 0.8, 0.25)
            self.viewer.add_geom(target1)

            self.sphere_obstacles = []
            for s in range(len(self.forbidden_spheres)):
                target1 = rendering.make_circle(radius=self.forbidden_spheres[s][1] * scale)
                sph_obs_trans = rendering.Transform(self.forbidden_spheres[s][0] * scale + offset)
                target1.add_attr(sph_obs_trans)
                target1.set_color(0.25, 0.25, 0.25)
                self.viewer.add_geom(target1)

                self.sphere_obstacles.append(sph_obs_trans)

            self.agent = agent

        if self.state is None: return None

        new_pos = self.state[0:2] * scale + offset
        self.agent_transform.set_translation(new_pos[0], new_pos[1])

        new_target = self.targets[0] * scale + offset
        self.target_transform.set_translation(new_target[0], new_target[1])

        if self.draw_perception:
            # draw target vec
            target_vec_end = (self.vec + self.state[0:2]) * scale + offset
            line = self.viewer.draw_line(new_pos, target_vec_end)
            line.attrs[1].stroke = 8
            line.attrs[0].vec4 = (1.0, 0.6, 0.0, 1.0)

            target_vec_end = (self.closest_vec[0:2] + self.state[0:2]) * scale + offset
            line = self.viewer.draw_line(new_pos, target_vec_end)
            line.attrs[1].stroke = 8
            line.attrs[0].vec4 = (0.0, 0.9, 0.0, 1.0)

            target_vec_end = (self.closest_vec[2:] + self.state[0:2]) * scale + offset
            line = self.viewer.draw_line(new_pos, target_vec_end)
            line.attrs[1].stroke = 8
            line.attrs[0].vec4 = (0.0, 0.4, 0.0, 1.0)

        if self.draw_path:
            if len(self.paths) > 2:
                for i in range(1, len(self.paths)):
                    start = self.paths[i][0][0:2] * scale + offset
                    end = self.paths[i][1][0:2] * scale + offset
                    line = self.viewer.draw_line(start, end)
                    line.attrs[1].stroke = 15
                    if self.paths[i][2] == 'TASK':
                        line.attrs[0].vec4 = (0.0, 1.0, 0.0, 1.0)
                    elif self.paths[i][2] == 'SAFE':
                        line.attrs[0].vec4 = (1.0, 0.0, 0.0, 1.0)

        for s in range(len(self.forbidden_spheres)):
            pos = self.forbidden_spheres[s][0] * scale + offset
            self.sphere_obstacles[s].set_translation(pos[0], pos[1])


        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def reset(self):
        if not self.randomize_initial_state or self.test_mode:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) * 0 + 0.0001
        else:
            self.state = self.np_random.uniform(low=-0.01, high=0.01, size=(4,))
            self.state[0] = self.np_random.uniform(-0.1, 0.1)
            self.state[1] = self.np_random.uniform(-0.1, 0.1)
        self.state += self.init_pose

        # self.forbidden_spheres = copy.deepcopy(self.forbidden_spheres_original)

        self.current_action = np.ones(2)

        if self.resample_MP and not self.test_mode:
            self.mass = np.random.uniform(self.mass_range[0], self.mass_range[1])

        if self.resample_task_on_reset:
            self.resample_task()

        if self.train_UP:
            obs = np.concatenate([self.state, [self.mass]*1])
            # obs = np.concatenate([self.state, [1.5]])
        else:
            obs = np.copy(self.state[2:])

        if self.input_target:
            obs = np.concatenate([obs, (self.targets[0] - self.state[0:2])])

        sphere_collide, closest_vec = self.in_forbidden_spheres()
        if self.input_target:
            obs = np.concatenate([obs, closest_vec])

        self.vec = self.targets[0] - self.state[0:2]

        self.closest_vec = closest_vec

        self.cur_step = 0

        self.current_mode = 'NORMAL'

        self.paths = []

        return obs

    def state_vector(self):
        return np.copy(self.state)

    def set_state_vector(self, s):
        self.state = np.copy(s)