__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env

# swing up and balance of double inverted pendulum
class DartDoubleInvertedPendulumEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0],[-1.0]])
        self.action_scale = 40
        self.avg_div = 2
        self.cur_step = 0
        obs_dim = 6

        self.dyn_models = [None]
        self.dyn_model_id = 0
        self.base_path = None
        self.transition_locator = None

        if self.avg_div > 1:
            obs_dim += self.avg_div

        dart_env.DartEnv.__init__(self, 'inverted_double_pendulum.skel', 2, obs_dim, self.control_bounds, dt=0.005, disableViewer=True)
        utils.EzPickle.__init__(self)

    def step(self, a):
        reward = 0.0

        clamped_control = np.copy(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = clamped_control[0] * self.action_scale

        state_act = np.concatenate([self.state_vector(), tau/self.action_scale])
        state_pre = np.copy(self.state_vector())
        if self.dyn_model_id == 0 or self.dyn_models[self.dyn_model_id-1] is None:
            self.do_simulation(tau, self.frame_skip)
        elif self.dyn_models[self.dyn_model_id-1] is not None and self.base_path is None:
            new_state = self.dyn_models[self.dyn_model_id-1].do_simulation(self.state_vector(), tau, self.frame_skip)
            self.set_state_vector(new_state)
        elif self.dyn_models[self.dyn_model_id-1] is not None:
            cur_state = self.state_vector()
            tau /= self.action_scale
            cur_act = tau
            if self.transition_locator is None:
                base_state_act = self.base_path['env_infos']['state_act'][self.cur_step]
                base_state = base_state_act[0:len(cur_state)]
                base_act = base_state_act[-len(cur_act):]
                base_next_state = base_state+self.base_path['env_infos']['next_state'][self.cur_step]
            else:
                query = self.transition_locator.kneighbors([np.concatenate([cur_state, cur_act])])
                dist = query[0][0][0]
                self.total_dist.append(dist)

                ind = query[1][0][0]
                base_state_act = self.transition_locator._fit_X[ind]
                if dist > 2.0:
                    print('----------------')
                    print(base_state_act, cur_state, cur_act, a)
                base_state = base_state_act[0:len(cur_state)]
                base_act = base_state_act[-len(cur_act):]
                base_next_state = base_state + self.transition_locator._y[ind]
            new_state = self.dyn_models[self.dyn_model_id-1].do_simulation_corrective(base_state, base_act, \
                                            self.frame_skip, base_next_state, cur_state - base_state, cur_act-base_act)
            new_state = base_next_state + 0.1 * (new_state - base_next_state)/np.linalg.norm(new_state - base_next_state)
            self.set_state_vector(new_state)

        ob = self._get_obs()

        reward -= 1.0 * np.abs(self.robot_skeleton.q[0])**2
        reward += np.cos(ob[1]) + np.cos(ob[2])
        if (np.cos(ob[1]) + np.cos(ob[2])) > 1.8:
            reward += 5

        notdone = np.isfinite(ob).all()
        done = not notdone# or abs(self.robot_skeleton.q[0]) > 2.0

        if self.dyn_model_id != 0:
            reward *= 1.0
        self.cur_step += 1
        if self.base_path is not None and self.dyn_model_id != 0 and self.transition_locator is None:
            if len(self.base_path['env_infos']['state_act']) <= self.cur_step:
                done = True

        #if self.dyn_model_id != 0 and len(self.base_path['env_infos']['state_act']) <= self.cur_step:
        #    print('Dist: ', self.total_dist)

        return ob, reward, done, {'state_act': state_act, 'next_state':self.state_vector()-state_pre, 'dyn_model_id':self.dyn_model_id}


    def _get_obs(self):
        state = np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()
        if self.avg_div > 1:
            return_state = np.zeros(len(state) + self.avg_div)
            return_state[0:len(state)] = state
            return_state[len(state) + self.state_index] = 1.0
            return return_state

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        if self.np_random.uniform(low=0, high=1, size=1) > 0.5:
            qpos[1] += np.pi
        else:
            qpos[1] += -np.pi

        self.state_index = self.dyn_model_id

        self.set_state(qpos, qvel)

        if self.base_path is not None:
            base_state = self.base_path['env_infos']['state_act'][0][0:len(self.state_vector())]
            self.set_state_vector(base_state + self.np_random.uniform(low=-0.01, high=0.01, size=len(base_state)))

        self.cur_step = 0
        self.total_dist = []

        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
