import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import pydart2.utils.transformations as transformations

class DartBridgePuzzle(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.targets = [np.array([0.0, 0.5, 0.0]), np.array([1.0, 0.5, 0.0]), np.array([2.0, 0.5, 0.0]), np.array([3.0, 0.5, 0.0]), np.array([4.0, 0.5, 0.0]),
                        np.array([4.7, 0.5, -0.7]), np.array([5.4, 0.5, -1.4]), np.array([6.1, 0.5, -2.1]), np.array([6.8, 0.5, -2.8]), np.array([7.5, 0.5, -3.5]),
                        np.array([8.33, 0.5, -2.8]), np.array([9.16, 0.5, -2.1]), np.array([10.0, 0.5, -1.4]), np.array([10.83, 0.5, -0.7]), np.array([11.66, 0.5, 0.0]), np.array([12.5, 0.5, 0.7]),
                        np.array([13.33, 0.5, 0.0]), np.array([14.16, 0.5, -0.7]), np.array([15.0, 0.5, -1.4]), np.array([15.83, 0.5, -2.1]), np.array([16.66, 0.5, -2.8]), np.array([17.5, 0.5, -3.5]),
                        np.array([18.33, 0.5, -2.8]), np.array([19.16, 0.5, -2.1]), np.array([20.0, 0.5, -1.4]), np.array([20.83, 0.5, -0.7]), np.array([21.66, 0.5, 0.0]), np.array([22.5, 0.5, 0.7]),
                        np.array([23.0, 0.5, 0.7]), np.array([23.5, 0.5, 0.7])]

        self.action_scale = 80

        self.vel_track = True
        self.target_vel = 5.0

        self.action_filtering = 0  # window size of filtering, 0 means no filtering
        self.action_filter_cache = []

        self.alive_bonus = 0.0

        self.hierarchical = False
        if self.hierarchical:
            self.h_horizon = 20
            self.action_scale = 2.0
            self.num_way_point = 1

        self.control_bounds = np.array([[1.0, 1.0], [-1.0, -1.0]])

        self.movement_offset = 0.0

        if self.hierarchical:
            self.control_bounds = np.array([[1.0] * (2*self.num_way_point), [-1.0] * (2*self.num_way_point)])

        dart_env.DartEnv.__init__(self, 'bridge_puzzle.skel', 2, 6, self.control_bounds, dt=0.01, disableViewer=True)

        #self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(5.0)
        for bn in self.dart_world.skeletons[1].bodynodes:
            bn.set_friction_coeff(0.3)
        for bn in self.dart_world.skeletons[-1].bodynodes:
            bn.set_friction_coeff(0.3)

        # self.set_task([0.2])

        self.add_perturbation = True
        self.perturbation_parameters = [0.02, 100, 0, 50]

        utils.EzPickle.__init__(self)

    def do_simulation(self, tau, n_frames):
        if self.add_perturbation:
            if self.perturbation_duration == 0:
                self.perturb_force *= 0
                if np.random.random() < self.perturbation_parameters[0]:
                    angle_rand = np.random.uniform(0, np.pi*2)
                    direction_rand = np.array([np.sin(angle_rand), 0.0, np.cos(angle_rand)])
                    self.perturb_force = direction_rand * self.perturbation_parameters[1]
                    self.perturbation_duration = self.perturbation_parameters[3]
            else:
                self.perturbation_duration -= 1

        for _ in range(n_frames):
            if self.add_perturbation:
                self.robot_skeleton.bodynodes[self.perturbation_parameters[2]].add_ext_force(self.perturb_force)

            self.robot_skeleton.set_forces(np.concatenate([[0,0,0], tau]))
            self.dart_world.step()

    def _rotate_bridge(self, angle):
        pos_rot_matrix = transformations.rotation_matrix(angle, [0, 1, 0])
        neg_rot_matrix = transformations.rotation_matrix(-angle, [0, 1, 0])
        for i in range(1, int(len(self.dart_world.skeletons[0].bodynodes[0].shapenodes)/2-2)):
            current_trans = self.dart_world.skeletons[0].bodynodes[0].shapenodes[i].relative_transform()
            current_trans[0:3, 0:3] = np.identity(3)
            current_trans = np.dot(current_trans, neg_rot_matrix if i % 2 == 0 else pos_rot_matrix)
            self.dart_world.skeletons[0].bodynodes[0].shapenodes[i].set_relative_transform(current_trans)
            self.dart_world.skeletons[0].bodynodes[0].shapenodes[int(i + len(self.dart_world.skeletons[0].bodynodes[0].shapenodes)/2)].set_relative_transform(current_trans)


    def resample_task(self):
        bridge_rotation = np.random.random() * 0.85
        self._rotate_bridge(bridge_rotation)
        return [bridge_rotation]

    def set_task(self, task_params):
        bridge_rotation = task_params[0]
        self._rotate_bridge(bridge_rotation)

    def step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        self.action_filter_cache.append(clamped_control)
        if len(self.action_filter_cache) > self.action_filtering:
            self.action_filter_cache.pop(0)
        if self.action_filtering > 0:
            clamped_control = np.mean(self.action_filter_cache, axis=0)

        if not self.hierarchical:
            tau = np.zeros(3)
            tau[[0,2]] = np.multiply(clamped_control, self.action_scale)
            vel = self.robot_skeleton.bodynodes[-1].dC
            # if np.linalg.norm(vel) > 4:
            #     tau *= 0.0
            tau[0] -= vel[0]*10.0
            tau[2] -= vel[2]*10.0

            if self.vel_track:
                target_dir = clamped_control / np.linalg.norm(clamped_control)

                ang = np.arctan2(target_dir[0], target_dir[1])

                assert(np.linalg.norm(np.array([np.sin(ang), np.cos(ang)]) - target_dir) < 1e-4)

                ang += self.movement_offset
                target_dir = np.array([np.sin(ang), np.cos(ang)])

                new_target_vel = target_dir * self.target_vel
                tau = np.zeros(3)
                # print(new_target_vel[[0,1]], vel[[0,2]])
                tau[0] = (new_target_vel[0] - vel[0]) * 200
                tau[2] = (new_target_vel[1] - vel[2]) * 200

            self.do_simulation(tau, self.frame_skip)
        else:
            prev_pos = self.robot_skeleton.bodynodes[-1].com()[[0,2]]
            scaled_a = a * self.action_scale
            targets = []
            for i in range(self.num_way_point):
                new_pos = prev_pos + scaled_a[i*2:i*2+2]
                targets.append(new_pos)
                prev_pos = np.copy(new_pos)

            for i in range(self.h_horizon):
                tau = np.zeros(3)
                tvec = targets[int(i / (self.h_horizon * 1.0 / self.num_way_point))] - self.robot_skeleton.bodynodes[-1].com()[[0,2]]
                tvec /= np.linalg.norm(tvec)
                tau[[0,2]] = tvec * 50
                self.do_simulation(tau, self.frame_skip)
            self.dart_world.skeletons[-2].q = np.array([targets[0][0], 0.0, targets[0][1]])


        ob = self._get_obs()

        vec = self.robot_skeleton.bodynodes[-1].com() - self.targets[self.current_target_id]

        # reward_dist = - np.linalg.norm(vec) * 0.65

        reward_ctrl = - np.square(a).sum()*0.05
        reward = reward_ctrl + 0.1

        if np.linalg.norm(vec) < self.max_rew_dist:
            reward += (self.max_rew_dist - np.linalg.norm(vec)) * 10
            self.max_rew_dist = np.linalg.norm(vec)

        if np.linalg.norm(vec) < 0.5:
            reward += 50
            self.current_target_id += 1
            if self.current_target_id >= len(self.targets):
                self.current_target_id -= 1
            self.dart_world.skeletons[-2].q = self.targets[self.current_target_id]

            self.max_rew_dist = np.linalg.norm(
                self.robot_skeleton.bodynodes[-1].com() - self.targets[self.current_target_id])

        # print(np.linalg.norm(self.robot_skeleton.bodynodes[-1].dC))

        #done = not (np.isfinite(s).all() and (-reward_dist > 0.02))
        done = False

        if self.robot_skeleton.bodynodes[-1].com()[1] < 0.35 or self.robot_skeleton.bodynodes[-1].com()[1] > 0.9:
            done = True
            reward -= 100

        return ob, reward, done, {'done_return':done}

    def _get_obs(self):
        pos = self.robot_skeleton.bodynodes[-1].C
        vel = self.robot_skeleton.bodynodes[-1].dC

        return np.array([pos[0], pos[2], vel[0], vel[2], self.dart_world.skeletons[-2].C[0], self.dart_world.skeletons[-2].C[2]])

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qpos[1] = 0.0
        qvel[1] = 0.0
        self.set_state(qpos, qvel)

        if not self.hierarchical:
            self.dart_world.skeletons[-2].q = np.array([1000, 10, 10])
        self.dart_world.skeletons[-2].q = self.targets[0]
        self.current_target_id = 0

        self.max_rew_dist = np.linalg.norm(self.robot_skeleton.bodynodes[-1].com() - self.targets[self.current_target_id])

        self.movement_offset = np.random.uniform(-0.3, 0.3)

        return self._get_obs()


    def viewer_setup(self):
        self.track_skeleton_id = 0
        self._get_viewer().scene.tb.trans[2] = -40.5
        self._get_viewer().scene.tb._set_theta(-60)

    def state_vector(self):
        state_data = {}
        state_data['q'] = np.array(self.robot_skeleton.q)
        state_data['dq'] = np.array(self.robot_skeleton.dq)
        state_data['max_rew_dist'] = self.max_rew_dist
        state_data['cur_target_id'] = self.current_target_id

        return state_data

    def set_state_vector(self, state_data):
        self.robot_skeleton.q = state_data['q']
        self.robot_skeleton.dq = state_data['dq']
        self.max_rew_dist = state_data['max_rew_dist']
        self.current_target_id = state_data['cur_target_id']
        self.dart_world.skeletons[-2].q = self.targets[self.current_target_id]

