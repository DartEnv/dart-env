import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartBallWalkerEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0, 1.0],[-1.0, -1.0]])
        self.action_scale = 40
        dart_env.DartEnv.__init__(self, 'simple_torso.skel', 4, 4, control_bounds, disableViewer=True)
        utils.EzPickle.__init__(self)

    # def head_track(self):
    #

    def do_simulation(self, tau, n_frames):
        targ_angle = np.copy(tau)
        # print(targ_angle)
        # targ_angle = [0.5, -1.5]
        for _ in range(n_frames):
            kph, kdh, kih = 150, 15, 40
            kpt, kdt, kit = 150, 15, 200

            hq = self.robot_skeleton.q[1]    # head angle, relative to torso
            dhq = self.robot_skeleton.dq[1]  # head angular velocity
            tq = self.robot_skeleton.q[0]    # torso angle
            dtq = self.robot_skeleton.dq[0]  # torso angular velocity

            self.accum_error += hq * self.sim_dt  # accumulated head-torso angle error

            tau_h = -kph * (hq + tq - targ_angle[1]) - kdh * dhq - kih * self.accum_error   # torque applied to head
            tau_t = -kpt * (tq - targ_angle[0]) - kdt * dtq + kit * self.accum_error        # torque applied to torso

            tau = np.clip([tau_t, tau_h], -self.action_scale, self.action_scale)


            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def step(self, a):
        #if abs(a[0]) > 1:
        #    a[0] = np.sign(a[0])
        clamp_a = np.clip(a, -1, 1)

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = clamp_a[0] * 1.5# * self.action_scale
        tau[1] = clamp_a[1] * 1.5# * self.action_scale

        if self.cur_step < 200:
            tau = [0.8, -1.3]
        elif self.cur_step < 500:
            tau = [0.8, 0.6]
        elif self.cur_step < 800:
            tau = [0.1, 0.6]
        else:
            tau = [0.1, -0.6]

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        self.cur_step += 1


        reward = 0

        done = False
        return ob, reward, done, {'dyn_model_id':0}


    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        self.accum_error = 0
        self.cur_step = 0

        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
