import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import pydart2.utils.transformations as transformations

class DartBallCircle(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.action_scale = 80

        self.target_vel = 2.0

        self.alive_bonus = 0.0

        self.control_bounds = np.array([[1.0, 1.0], [-1.0, -1.0]])

        self.max_time = 15.0
        self.current_time = 0.0
        self.circle_radius = 6.0
        self.desired_rot_rate = 2 * np.pi / self.max_time

        dart_env.DartEnv.__init__(self, 'ball_circle.skel', 2, 6, self.control_bounds, dt=0.01, disableViewer=True)

        #self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(5.0)
        for bn in self.dart_world.skeletons[1].bodynodes:
            bn.set_friction_coeff(0.3)
        for bn in self.dart_world.skeletons[-1].bodynodes:
            bn.set_friction_coeff(0.3)

        self.add_perturbation = True
        self.perturbation_parameters = [0.02, 100, 0, 50]

        utils.EzPickle.__init__(self)

    def step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        vel = self.robot_skeleton.bodynodes[-1].dC

        target_dir = clamped_control / np.linalg.norm(clamped_control)

        ang = np.arctan2(target_dir[0], target_dir[1]) + self.movement_offset

        target_dir = np.array([np.sin(ang), np.cos(ang)])

        new_target_vel = target_dir * self.target_vel
        tau = np.zeros(3)
        tau[0] = (new_target_vel[0] - vel[0]) * 200
        tau[2] = (new_target_vel[1] - vel[2]) * 200

        self.do_simulation(tau, self.frame_skip)

        ob = self._get_obs()

        self.current_time += self.dt
        elapsed_angle = self.desired_rot_rate * self.current_time

        self.target = np.array([-np.sin(elapsed_angle) * self.circle_radius, np.cos(elapsed_angle) * self.circle_radius])
        self.dart_world.skeletons[-2].q = np.array([self.target[0], 0.5, self.target[1]])
        vec = np.array([self.target[0] - self.robot_skeleton.C[0], self.target[1] - self.robot_skeleton.C[2]])

        reward_dist = - np.linalg.norm(vec) * 0.1

        reward_ctrl = - np.square(a).sum()*0.05

        zone_penalty = 0.0
        if self.robot_skeleton.C[0] < -self.circle_radius * 0.6 or self.robot_skeleton.C[0] > self.circle_radius * 0.6:
            zone_penalty = -10.0

        reward = reward_dist + reward_ctrl + zone_penalty

        done = self.current_time > self.max_time


        return ob, reward, done, {'done_return':done}

    def _get_obs(self):
        pos = self.robot_skeleton.bodynodes[-1].C
        vel = self.robot_skeleton.bodynodes[-1].dC

        return np.array([pos[0], pos[2], vel[0], vel[2], self.target[0], self.target[1]])

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qpos[1] = 0.0
        qvel[1] = 0.0
        qpos[2] += self.circle_radius
        self.set_state(qpos, qvel)

        self.target = np.array([0.0, self.circle_radius])

        self.current_time = 0.0

        self.movement_offset = np.random.uniform(-0.3, 0.3)
        self.target_vel = np.random.uniform(1.0, 2.5)

        return self._get_obs()


    def viewer_setup(self):
        self.track_skeleton_id = 0
        self._get_viewer().scene.tb.trans[2] = -20.5
        self._get_viewer().scene.tb._set_theta(-60)

    def state_vector(self):
        state_data = {}
        state_data['q'] = np.array(self.robot_skeleton.q)
        state_data['dq'] = np.array(self.robot_skeleton.dq)
        state_data['current_time'] = self.current_time

        return state_data

    def set_state_vector(self, state_data):
        self.robot_skeleton.q = state_data['q']
        self.robot_skeleton.dq = state_data['dq']
        self.current_time = state_data['current_time']

