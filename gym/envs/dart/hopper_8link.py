import os
import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartHopper8LinkEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        self.action_scale = 50
        self.include_action_in_obs = False
        self.randomize_dynamics = False
        obs_dim = 19

        if self.include_action_in_obs:
            obs_dim += len(self.control_bounds[0])
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        self.supp_input = True

        self.reverse_order = False

        self.feet_specialized = False

        self.pid_controller = None#[150, 9]
        if self.pid_controller is not None:
            self.torque_limit = [-100, 100]
            self.action_scale = 1.0

            # MMHACK: PCA action space
            # self.action_scale = 4.0
            # self.control_bounds = np.array([[1.0]*6, [-1.0]*6])
            # import pickle
            # fullpath = os.path.join(os.path.dirname(__file__), "models", "pcamodel_8link.pkl")
            # self.pca = pickle.load(open(fullpath, 'rb'))

        if self.supp_input:
            obs_dim += 3 * 8  # [contact, local_x, local_y]

        dart_env.DartEnv.__init__(self, 'hopper_multilink/hopperid_8link.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        if self.randomize_dynamics:
            self.bodynode_original_masses = []
            self.bodynode_original_frictions = []
            for bn in self.robot_skeleton.bodynodes:
                self.bodynode_original_masses.append(bn.mass())
                self.bodynode_original_frictions.append(bn.friction_coeff())

        self.dart_world.set_collision_detector(3)

        self.initialize_articunet()

        if self.pid_controller is not None:
            self.joint_lower_lim = np.array(self.robot_skeleton.q_lower)[3:]
            self.joint_upper_lim = np.array(self.robot_skeleton.q_upper)[3:]
            expand = (self.joint_upper_lim - self.joint_lower_lim) * 0.1
            self.joint_lower_lim -= expand
            self.joint_upper_lim += expand

        utils.EzPickle.__init__(self)

    def initialize_articunet(self, supp_input = None, reverse_order = None, feet_specialized = None):
        self.supp_input = supp_input if supp_input is not None else self.supp_input
        self.reverse_order = reverse_order if reverse_order is not None else self.reverse_order
        self.feet_specialized = feet_specialized if feet_specialized is not None else self.feet_specialized
        # setups for articunet
        self.state_dim = 32
        self.enc_net = []
        self.act_net = []
        self.vf_net = []
        self.merg_net = []
        self.net_modules = []
        self.net_vf_modules = []
        if self.include_action_in_obs:
            self.enc_net.append([self.state_dim, 5, 64, 1, 'planar_enc'])
            self.enc_net.append([self.state_dim, 3, 64, 1, 'revolute_enc'])
        elif self.supp_input:
            self.enc_net.append([self.state_dim, 5 + 3, 64, 1, 'planar_enc'])
            self.enc_net.append([self.state_dim, 2 + 3, 64, 1, 'revolute_enc'])
        else:
            self.enc_net.append([self.state_dim, 5, 64, 1, 'planar_enc'])
            self.enc_net.append([self.state_dim, 2, 64, 1, 'revolute_enc'])

        self.enc_net.append([self.state_dim, 5, 64, 1, 'vf_planar_enc'])
        if not self.include_action_in_obs:
            self.enc_net.append([self.state_dim, 2, 64, 1, 'vf_revolute_enc'])
        else:
            self.enc_net.append([self.state_dim, 3, 64, 1, 'vf_revolute_enc'])

        # specialize ankle joint
        self.enc_net.append([self.state_dim, 2, 64, 1, 'ankle_enc'])

        self.act_net.append([self.state_dim, 1, 64, 1, 'revolute_act'])

        # specialize ankle joint
        self.act_net.append([self.state_dim, 1, 64, 1, 'ankle_act'])

        self.vf_net.append([self.state_dim, 1, 64, 1, 'vf_out'])
        self.merg_net.append([self.state_dim, 1, 64, 1, 'merger'])

        # 4 - 5, 5 - 7, 6 - 8

        # value function modules
        if not self.include_action_in_obs:
            self.net_vf_modules.append([[8, 18], 3, None])
            self.net_vf_modules.append([[7, 17], 3, [0]])
            self.net_vf_modules.append([[6, 16], 3, [1]])
            self.net_vf_modules.append([[5, 15], 3, [2]])
            self.net_vf_modules.append([[4, 14], 3, [3]])
            self.net_vf_modules.append([[3, 13], 3, [4]])
            self.net_vf_modules.append([[2, 12], 3, [5]])
        else:
            self.net_vf_modules.append([[8, 18, 25], 3, None])
            self.net_vf_modules.append([[7, 17, 24], 3, [0]])
            self.net_vf_modules.append([[6, 16, 23], 3, [1]])
            self.net_vf_modules.append([[5, 15, 22], 3, [2]])
            self.net_vf_modules.append([[4, 14, 21], 3, [3]])
            self.net_vf_modules.append([[3, 13, 20], 3, [4]])
            self.net_vf_modules.append([[2, 12, 19], 3, [5]])
        self.net_vf_modules.append([[0, 1, 9, 10, 11], 2, [6]])
        self.net_vf_modules.append([[], 7, [7]])

        # policy modules
        if not self.reverse_order:
            self.net_modules.append([[8, 18], 1 if not self.feet_specialized else 4, None])
            self.net_modules.append([[7, 17], 1 if not self.feet_specialized else 4, [0]])
            self.net_modules.append([[6, 16], 1, [1]])
            self.net_modules.append([[5, 15], 1, [2]])
            self.net_modules.append([[4, 14], 1, [3]])
            self.net_modules.append([[3, 13], 1, [4]])
            self.net_modules.append([[2, 12], 1, [5]])
            self.net_modules.append([[0, 1, 9, 10, 11], 0, [6]])

            if self.include_action_in_obs:
                self.net_modules[0][0] += [25]
                self.net_modules[1][0] += [24]
                self.net_modules[2][0] += [23]
                self.net_modules[3][0] += [22]
                self.net_modules[4][0] += [21]
                self.net_modules[5][0] += [20]
                self.net_modules[6][0] += [19]
            elif self.supp_input:
                self.net_modules[0][0] += [40, 41, 42]
                self.net_modules[1][0] += [37, 38, 39]
                self.net_modules[2][0] += [34, 35, 36]
                self.net_modules[3][0] += [31, 32, 33]
                self.net_modules[4][0] += [28, 29, 30]
                self.net_modules[5][0] += [25, 26, 27]
                self.net_modules[6][0] += [22, 23, 24]
                self.net_modules[7][0] += [19, 20, 21]

            self.net_modules.append([[], 8, [7, 6], None, False])
            self.net_modules.append([[], 8, [7, 5], None, False])
            self.net_modules.append([[], 8, [7, 4], None, False])
            self.net_modules.append([[], 8, [7, 3], None, False])
            self.net_modules.append([[], 8, [7, 2], None, False])
            self.net_modules.append([[], 8, [7, 1], None, False])
            self.net_modules.append([[], 8, [7, 0], None, False])

            self.net_modules.append([[], 5, [8]])
            self.net_modules.append([[], 5, [9]])
            self.net_modules.append([[], 5, [10]])
            self.net_modules.append([[], 5, [11]])
            self.net_modules.append([[], 5, [12]])
            self.net_modules.append([[], 5 if not self.feet_specialized else 6, [13]])
            self.net_modules.append([[], 5 if not self.feet_specialized else 6, [14]])

            self.net_modules.append([[], None, [15, 16, 17, 18, 19, 20, 21], None, False])

    def do_simulation(self, tau, n_frames):
        if self.pid_controller is not None:
            target_angles = (np.copy(tau[3:]) + 1.0) / 2.0 * (self.joint_upper_lim - self.joint_lower_lim) + self.joint_lower_lim
        average_torque = []
        for _ in range(n_frames):
            if self.pid_controller is not None:
                jpos = np.array(self.robot_skeleton.q)[3:]
                jvel = np.array(self.robot_skeleton.dq)[3:]

                torque = self.pid_controller[0] * (target_angles - jpos) - self.pid_controller[1] * jvel
                clipped_torque = np.clip(torque, self.torque_limit[0], self.torque_limit[1])

                tau = np.concatenate([[0.0] * 3, clipped_torque])
                average_torque.append(clipped_torque)

            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()
        if self.pid_controller is not None:
            self.average_torque = np.mean(average_torque, axis=0)

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        if self.include_action_in_obs:
            self.prev_a = np.copy(clamped_control)
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        # MMHACK: PCA action space
        # import pdb; pdb.set_trace()
        # tau[3:] = self.pca.inverse_transform([clamped_control * self.action_scale])[0]

        self.do_simulation(tau, self.frame_skip)

    def step(self, a):
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.q[0]
        self.advance(a)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        fall_on_ground = False
        contacts = self.dart_world.collision_result.contacts
        if self.supp_input:
            self.body_contact_list *= 0
        for contact in contacts:
            if contact.bodynode1.skid == 1 and contact.bodynode1.id < len(self.robot_skeleton.bodynodes) - 3:
                fall_on_ground = True
            if contact.bodynode2.skid == 1 and contact.bodynode2.id < len(self.robot_skeleton.bodynodes) - 3:
                fall_on_ground = True
            if self.supp_input:
                for bid, bn in enumerate(self.robot_skeleton.bodynodes):
                    if bid >= 2:
                        if contact.bodynode1 == bn or contact.bodynode2 == bn:
                            self.body_contact_list[bid - 2] = 1.0

        alive_bonus = 1.0
        reward = 0.3 * (posafter - posbefore) / self.dt
        reward += alive_bonus
        if self.pid_controller is None:
            reward -= 2e-3 * np.square(a).sum()
        else:
            reward -= 1e-3 * np.square(self.average_torque / self.torque_limit[1]).sum()

        # reward -= 3e-3 * np.abs(np.dot(a, self.robot_skeleton.dq[3:])).sum()
        # penalize distance between whole-body COM and foot COM
        # reward -= 1e-4 * np.square(self.robot_skeleton.bodynodes[2].C - self.robot_skeleton.bodynodes[-1].C).sum()

        s = self.state_vector()
        self.accumulated_rew += reward
        self.num_steps += 1.0
        #print(self.num_steps)
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > 1.7) and (height < self.init_height + 0.5))
        if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
            reward = 0
        if fall_on_ground:
            done = True
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.include_action_in_obs:
            state = np.concatenate([state, self.prev_a])

        if self.supp_input:
            for i, bn in enumerate(self.robot_skeleton.bodynodes):
                if i >= 2:
                    com_off = bn.C - self.robot_skeleton.bodynodes[2].C
                    state = np.concatenate([state, [self.body_contact_list[i-2], com_off[0], com_off[1]]])

        return state


    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        if self.supp_input:
            self.body_contact_list = np.zeros(len(self.robot_skeleton.bodynodes) - 2)

        state = self._get_obs()

        self.init_height = self.robot_skeleton.bodynodes[2].com()[1]

        if self.include_action_in_obs:
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        self.accumulated_rew = 0.0
        self.num_steps = 0.0

        if self.randomize_dynamics:
            for i in range(len(self.robot_skeleton.bodynodes)):
                self.robot_skeleton.bodynodes[i].set_mass(
                    self.bodynode_original_masses[i] + np.random.uniform(-1.5, 1.5))
                self.robot_skeleton.bodynodes[i].set_friction_coeff(
                    self.bodynode_original_frictions[i] + np.random.uniform(-0.5, 0.5))

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5

    def state_vector(self):
        s = np.copy(np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]))
        s[1] += self.init_height
        return s

    def set_state_vector(self, s):
        snew = np.copy(s)
        snew[1] -= self.init_height
        self.robot_skeleton.q = snew[0:len(self.robot_skeleton.q)]
        self.robot_skeleton.dq = snew[len(self.robot_skeleton.q):]