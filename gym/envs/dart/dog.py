__author__ = 'yuwenhao'


import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *


class DartDogEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*1,[-1.0]*1])
        self.action_scale = 200
        obs_dim = 11

        dart_env.DartEnv.__init__(self, 'dog/dog.skel', 15, obs_dim, self.control_bounds, disableViewer=False)

        self.dart_world.set_collision_detector(3) # 3 is ode collision detector

        utils.EzPickle.__init__(self)

    def _step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)

        posbefore = self.robot_skeleton.bodynodes[0].com()[0]
        self.do_simulation(tau, self.frame_skip)
        posafter = self.robot_skeleton.bodynodes[0].com()[0]
        height = self.robot_skeleton.bodynodes[0].com()[1]
        side_deviation = abs(self.robot_skeleton.bodynodes[0].com()[2])

        alive_bonus = 1.0
        reward = 0.6*(posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (height < 1.8) and (side_deviation < .4))
        ob = self._get_obs()

        return ob, reward, done, {'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(a).sum(), 'done_return':done}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq
        ])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -3.5
