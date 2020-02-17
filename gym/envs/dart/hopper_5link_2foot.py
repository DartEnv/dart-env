import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartHopper5Link2FootEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, -1.0]])
        self.action_scale = 100
        obs_dim = 15

        dart_env.DartEnv.__init__(self, 'hopper_multilink/hopperid_5link_2foot.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        self.dart_world.set_collision_detector(3)

        # setups for articunet
        self.state_dim = 32
        self.enc_net = []
        self.act_net = []
        self.net_modules = []

        self.enc_net.append([self.state_dim, 5, 64, 1, 'planar_enc'])
        self.enc_net.append([self.state_dim, 2, 64, 1, 'revolute_enc'])
        self.act_net.append([self.state_dim, 1, 64, 1, 'revolute_act'])

        self.net_modules.append([[6, 14], 1, None])
        self.net_modules.append([[5, 13], 1, None])
        self.net_modules.append([[4, 12], 1, [0, 1]])
        self.net_modules.append([[3, 11], 1, [2]])
        self.net_modules.append([[2, 10], 1, [3]])
        self.net_modules.append([[0, 1, 7, 8, 9], 0, [4]])

        self.net_modules.append([[], 2, [5, 4]])
        self.net_modules.append([[], 2, [5, 3]])
        self.net_modules.append([[], 2, [5, 2]])
        self.net_modules.append([[], 2, [5, 1]])
        self.net_modules.append([[], 2, [5, 0]])

        self.net_modules.append([[], None, [6, 7, 8, 9, 10], None, False])

        utils.EzPickle.__init__(self)

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def step(self, a):
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.q[0]
        self.advance(a)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        self.accumulated_rew += reward
        self.num_steps += 1.0
        #print(self.num_steps)
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > self.init_height - 0.4) and (height < self.init_height + 0.5) and (abs(ang) < .4))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        return state


    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        state = self._get_obs()

        self.init_height = self.robot_skeleton.bodynodes[2].com()[1]

        self.accumulated_rew = 0.0
        self.num_steps = 0.0

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5