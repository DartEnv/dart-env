import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.action_filter import *

class DartHumanoidBalanceEnv(dart_env.DartEnv, utils.EzPickle):
    # initialization
    def __init__(self):
        # define action space, humanoid has 23 actuated dofs
        control_bounds = np.array([[1.0] * 23, [-1.0] * 23])

        # set observation dimension, humanoid has (23+6)*2
        observation_dim = 56

        self.butterworth_filter = True

        # initialize dart-env
        dart_env.DartEnv.__init__(self, 'kima/kima_human_balance.skel', frame_skip=15, observation_size=observation_dim,
                                  action_bounds=control_bounds, dt=0.002)

        # enable self-collision check
        self.robot_skeleton.set_self_collision_check(True)

        # define the perturbation model
        self.push_direction = np.array([1, 0, 0])
        self.push_strength = 100.0
        self.push_target = 'head'

        self.current_step = 0

        if self.butterworth_filter:
            self.action_filter = ActionFilter(23, 3, int(1.0/self.dt), low_cutoff=0.0, high_cutoff=6.0)

        utils.EzPickle.__init__(self)

    # overload the do_simulation function to add perturbation
    def do_simulation(self, tau, n_frames):
        key_pressed = False
        if self._get_viewer() is not None:
            if hasattr(self._get_viewer(), 'key_being_pressed'):
                if self._get_viewer().key_being_pressed is not None:
                    key_pressed = True
        for _ in range(n_frames):
            if self.current_step < 30 or key_pressed:
                self.robot_skeleton.bodynode(self.push_target).add_ext_force(self.push_direction * self.push_strength)
                push_center = self.robot_skeleton.bodynode('head').com()
                self.dart_world.arrows = [[push_center, push_center + self.push_direction * self.push_strength * 0.005]]
            else:
                self.dart_world.arrows = []
            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    # take a step forward in time by executing action
    def step(self, action):
        # advance the simulation
        if self.butterworth_filter:
            action = self.action_filter.filter_action(action)

        tau = np.zeros(29)
        tau[6:] = np.clip(action, -1, 1)*200
        self.do_simulation(tau, self.frame_skip)

        # calculate reward
        reward = 3.0 - np.square(action).sum() * 0.1

        current_q = self.robot_skeleton.q
        q_dev = np.exp(-np.square(np.clip(current_q, -10, 10)).sum())
        reward += q_dev

        # termination criteria
        s = self.state_vector()
        height = self.robot_skeleton.bodynode('head').com()[1]
        done = not (np.isfinite(s).all() and (np.abs(s) < 100).all() and
                    (height - self.init_height > -0.35) and
                    np.abs(self.robot_skeleton.q[5]) < 1.0 and np.abs(self.robot_skeleton.q[4]) < 1.0 and
                    np.abs(self.robot_skeleton.q[3]) < 1.0)

        # get next observation
        observation = np.zeros(56)
        observation[0] = self.state_vector()[1]
        observation[1:] = self.state_vector()[3:]

        self.current_step += 1

        return observation, reward, done, {}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[[1]],
            self.robot_skeleton.q[3:],
            self.robot_skeleton.dq,
        ])

        return state

    # reset the rollout
    def reset_model(self):
        self.dart_world.reset()

        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        self.init_height = self.robot_skeleton.bodynode('head').com()[1]

        self.push_direction = np.array([np.random.uniform(-1, 1), 0.0, np.random.uniform(-1, 1)])
        self.push_direction /= np.linalg.norm(self.push_direction)
        self.push_strength = (np.random.random() + 1.0) * 100

        self.current_step = 0

        observation = np.zeros(56)
        observation[0] = self.state_vector()[1]
        observation[1:] = self.state_vector()[3:]

        if self.butterworth_filter:
            self.action_filter.reset_filter(np.zeros(23))

        return observation

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -3.5




