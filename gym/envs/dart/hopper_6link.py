import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartHopper6LinkEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        act_dim = 5
        self.state_dim = 32
        obs_dim = 15

        self.bidirectional_artnet = True

        self.latent_feedback = False
        if self.latent_feedback:
            act_dim += self.state_dim
            self.latent_obs = np.zeros(self.state_dim)
            obs_dim += self.state_dim

        self.control_bounds = np.array([[1.0] * act_dim,[-1.0] * act_dim])
        self.action_scale = 100
        self.include_action_in_obs = False
        self.randomize_dynamics = False

        if self.include_action_in_obs:
            obs_dim += len(self.control_bounds[0])
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        self.supp_input = True

        self.reverse_order = False

        self.feet_specialized = False

        if self.supp_input:
            obs_dim += 7 * 6  # [contact, local_x, local_y, rot]

        dart_env.DartEnv.__init__(self, 'hopper_multilink/hopperid_6link.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        if self.randomize_dynamics:
            self.bodynode_original_masses = []
            self.bodynode_original_frictions = []
            for bn in self.robot_skeleton.bodynodes:
                self.bodynode_original_masses.append(bn.mass())
                self.bodynode_original_frictions.append(bn.friction_coeff())

        self.dart_world.set_collision_detector(3)

        self.initialize_articunet()

        utils.EzPickle.__init__(self)

    def initialize_articunet(self, supp_input = None, reverse_order = None, feet_specialized = None):
        self.supp_input = supp_input if supp_input is not None else self.supp_input
        self.reverse_order = reverse_order if reverse_order is not None else self.reverse_order
        self.feet_specialized = feet_specialized if feet_specialized is not None else self.feet_specialized
        # setups for articunet

        self.enc_net = []
        self.act_net = []
        self.vf_net = []
        self.merg_net = []
        self.net_modules = []
        self.net_vf_modules = []
        self.generic_modules = []
        if self.include_action_in_obs:
            self.enc_net.append([self.state_dim, 5, 128, 1, 'planar_enc'])
            self.enc_net.append([self.state_dim, 3, 128, 1, 'revolute_enc'])
        elif self.supp_input:
            self.enc_net.append([self.state_dim, 5 + 7, 64, 1, 'planar_enc'])
            self.enc_net.append([self.state_dim, 2 + 7, 64, 1, 'revolute_enc'])
        else:
            self.enc_net.append([self.state_dim, 5, 64, 1, 'planar_enc'])
            self.enc_net.append([self.state_dim, 2, 64, 1, 'revolute_enc']) # include root velocity

        self.enc_net.append([self.state_dim, 5, 64, 1, 'vf_planar_enc'])
        if not self.include_action_in_obs:
            self.enc_net.append([self.state_dim, 2, 64, 1, 'vf_revolute_enc'])
        else:
            self.enc_net.append([self.state_dim, 3, 64, 1, 'vf_revolute_enc'])

        # specialize ankle joint
        self.enc_net.append([self.state_dim, 2, 64, 1, 'ankle_enc'])

        self.act_net.append([self.state_dim, 1, 32, 1, 'revolute_act'])

        # specialize ankle joint
        self.act_net.append([self.state_dim, 1, 64, 1, 'ankle_act'])

        self.vf_net.append([self.state_dim, 1, 64, 1, 'vf_out'])
        self.merg_net.append([self.state_dim, 1, 64, 1, 'merger'])

        # 4 - 5, 5 - 7, 6 - 8

        # value function modules
        if not self.include_action_in_obs:
            self.net_vf_modules.append([[6, 14], 3, None])
            self.net_vf_modules.append([[5, 13], 3, [0]])
            self.net_vf_modules.append([[4, 12], 3, [1]])
            self.net_vf_modules.append([[3, 11], 3, [2]])
            self.net_vf_modules.append([[2, 10], 3, [3]])
        else:
            self.net_vf_modules.append([[6, 14, 19], 3, None])
            self.net_vf_modules.append([[5, 13, 18], 3, [0]])
            self.net_vf_modules.append([[4, 12, 17], 3, [1]])
            self.net_vf_modules.append([[3, 11, 16], 3, [2]])
            self.net_vf_modules.append([[2, 10, 15], 3, [3]])
        self.net_vf_modules.append([[0, 1, 7, 8, 9], 2, [4]])
        self.net_vf_modules.append([[], 7, [5]])

        # policy modules
        if not self.reverse_order:
            self.net_modules.append([[6, 14], 1 if not self.feet_specialized else 4, None])
            self.net_modules.append([[5, 13], 1 if not self.feet_specialized else 4, [0]])
            self.net_modules.append([[4, 12], 1, [1]])
            self.net_modules.append([[3, 11], 1, [2]])
            self.net_modules.append([[2, 10], 1, [3]])
            self.net_modules.append([[0, 1, 7, 8, 9], 0, [4]])

            if self.include_action_in_obs:
                self.net_modules[0][0] += [19]
                self.net_modules[1][0] += [18]
                self.net_modules[2][0] += [17]
                self.net_modules[3][0] += [16]
                self.net_modules[4][0] += [15]
            elif self.supp_input:
                self.net_modules[0][0] += [50, 51, 52, 53, 54, 55, 56]
                self.net_modules[1][0] += [43, 44, 45, 46, 47, 48, 49]
                self.net_modules[2][0] += [36, 37, 38, 39, 40, 41, 42]
                self.net_modules[3][0] += [29, 30, 31, 32, 33, 34, 35]
                self.net_modules[4][0] += [22, 23, 24, 25, 26, 27, 28]
                self.net_modules[5][0] += [15, 16, 17, 18, 19, 20, 21]

            if self.bidirectional_artnet:
                self.net_modules.append([[2, 10], 1, [5]])
                self.net_modules.append([[3, 11], 1, [6]])
                self.net_modules.append([[4, 12], 1, [7]])
                self.net_modules.append([[5, 13], 1 if not self.feet_specialized else 4, [8]])
                self.net_modules.append([[6, 14], 1 if not self.feet_specialized else 4, [9]])
                if self.supp_input:
                    self.net_modules[6][0] += [22, 23, 24, 25, 26, 27, 28]
                    self.net_modules[7][0] += [29, 30, 31, 32, 33, 34, 35]
                    self.net_modules[8][0] += [36, 37, 38, 39, 40, 41, 42]
                    self.net_modules[9][0] += [43, 44, 45, 46, 47, 48, 49]
                    self.net_modules[10][0] += [50, 51, 52, 53, 54, 55, 56]
                self.net_modules.append([[], 8, [6, 4], None, False])
                self.net_modules.append([[], 8, [7, 3], None, False])
                self.net_modules.append([[], 8, [8, 2], None, False])
                self.net_modules.append([[], 8, [9, 1], None, False])
                self.net_modules.append([[], 8, [10, 0], None, False])

                self.net_modules.append([[], 5, [11]])
                self.net_modules.append([[], 5, [12]])
                self.net_modules.append([[], 5, [13]])
                self.net_modules.append([[], 5 if not self.feet_specialized else 6, [14]])
                self.net_modules.append([[], 5 if not self.feet_specialized else 6, [15]])

                self.net_modules.append([[], None, [16, 17, 18, 19, 20], None, False])

            else:
                self.net_modules.append([[], 8, [5, 4], None, False])
                self.net_modules.append([[], 8, [5, 3], None, False])
                self.net_modules.append([[], 8, [5, 2], None, False])
                self.net_modules.append([[], 8, [5, 1], None, False])
                self.net_modules.append([[], 8, [5, 0], None, False])

                self.net_modules.append([[], 5, [6]])
                self.net_modules.append([[], 5, [7]])
                self.net_modules.append([[], 5, [8]])
                self.net_modules.append([[], 5 if not self.feet_specialized else 6, [9]])
                self.net_modules.append([[], 5 if not self.feet_specialized else 6, [10]])

                self.net_modules.append([[], None, [11, 12, 13, 14, 15], None, False])
                if self.latent_feedback:
                    self.net_modules[-1][2] += [5]
        else:
            self.net_modules.append([[0, 1, 7, 8, 9], 0, None])
            self.net_modules.append([[2, 10], 1, [0]])
            self.net_modules.append([[3, 11], 1, [1]])
            self.net_modules.append([[4, 12], 1, [2]])
            self.net_modules.append([[5, 13], 1, [3]])
            self.net_modules.append([[6, 14], 1, [4]])

            self.net_modules.append([[], 8, [5, 1], None, False])
            self.net_modules.append([[], 8, [5, 2], None, False])
            self.net_modules.append([[], 8, [5, 3], None, False])
            self.net_modules.append([[], 8, [5, 4], None, False])

            self.net_modules.append([[], 5, [6]])
            self.net_modules.append([[], 5, [7]])
            self.net_modules.append([[], 5, [8]])
            self.net_modules.append([[], 5, [9]])
            self.net_modules.append([[], 5, [5]])

            self.net_modules.append([[], None, [10, 11, 12, 13, 14], None, False])

        # dynamic model
        self.dyn_enc_net = []
        self.dyn_act_net = []  # using actor as decoder
        self.dyn_merg_net = []
        self.dyn_net_modules = []
        self.dyn_enc_net.append([self.state_dim, 6, 256, 1, 'dyn_planar_enc'])
        self.dyn_enc_net.append([self.state_dim, 3, 256, 1, 'dyn_revolute_enc'])
        self.dyn_act_net.append([self.state_dim, 2, 256, 1, 'dyn_planar_dec'])
        self.dyn_act_net.append([self.state_dim, 6, 256, 1, 'dyn_revolute_dec'])
        self.dyn_merg_net.append([self.state_dim, 1, 256, 1, 'dyn_merger'])
        self.dyn_net_modules.append([[7, 15, 20], 1, None])
        self.dyn_net_modules.append([[6, 14, 19], 1, [0]])
        self.dyn_net_modules.append([[5, 13, 18], 1, [1]])
        self.dyn_net_modules.append([[4, 12, 17], 1, [2]])
        self.dyn_net_modules.append([[3, 11, 16], 1, [3]])
        self.dyn_net_modules.append([[0, 1, 2, 8, 9, 10], 0, [4]])

        self.dyn_net_modules.append([[], 4, [5, 4], None, False])
        self.dyn_net_modules.append([[], 4, [5, 3], None, False])
        self.dyn_net_modules.append([[], 4, [5, 2], None, False])
        self.dyn_net_modules.append([[], 4, [5, 1], None, False])
        self.dyn_net_modules.append([[], 4, [5, 0], None, False])

        self.dyn_net_modules.append([[], 2, [5]])
        self.dyn_net_modules.append([[], 3, [6]])
        self.dyn_net_modules.append([[], 3, [7]])
        self.dyn_net_modules.append([[], 3, [8]])
        self.dyn_net_modules.append([[], 3, [9]])
        self.dyn_net_modules.append([[], 3, [10]])
        self.dyn_net_modules.append([[], None, [11, 12, 13, 14, 15, 16], None, False])
        self.dyn_net_reorder = np.array([0, 1, 2, 6, 8, 10, 12, 14, 3, 4, 5, 7, 9, 11, 13, 15], dtype=np.int32)

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

        self.do_simulation(tau, self.frame_skip)

    def step(self, a):
        if self.latent_feedback:
            self.latent_obs = a[-self.state_dim:]
            a = a[0:len(a) - self.state_dim]
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.q[0]
        self.advance(a)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        fall_on_ground = False
        contacts = self.dart_world.collision_result.contacts
        if self.supp_input:
            self.body_contact_list *= 0.0
        for contact in contacts:
            if contact.bodynode1.skid == 1 and contact.bodynode1.id < len(self.robot_skeleton.bodynodes) - 2:
                fall_on_ground = True
            if contact.bodynode2.skid == 1 and contact.bodynode2.id < len(self.robot_skeleton.bodynodes) - 2:
                fall_on_ground = True
            if self.supp_input:
                for bid, bn in enumerate(self.robot_skeleton.bodynodes):
                    if bid >= 2:
                        if contact.bodynode1 == bn or contact.bodynode2 == bn:
                            self.body_contact_list[bid - 2] = 1.0

        alive_bonus = 1.0
        reward = 1.0*(posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        #reward -= 3e-3 * np.abs(np.dot(a, self.robot_skeleton.dq[3:])).sum()

        # penalize distance between whole-body COM and foot COM
        #reward -= 1e-4 * np.square(self.robot_skeleton.bodynodes[2].C - self.robot_skeleton.bodynodes[-1].C).sum()

        s = self.state_vector()
        self.accumulated_rew += reward
        self.num_steps += 1.0
        #print(self.num_steps)
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                     (height > self.init_height - 0.4) and (height < self.init_height + 0.5))
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
            rot = 0
            for i, bn in enumerate(self.robot_skeleton.bodynodes):
                if i >= 2:
                    rot += self.robot_skeleton.q[i]
                    #com_off = bn.C - self.robot_skeleton.C
                    com_off = bn.C - self.robot_skeleton.bodynodes[2].C # offset from root
                    com_vel = bn.dC
                    ang_vel = self.robot_skeleton.dq[i]
                    state = np.concatenate([state, [self.body_contact_list[i-2], com_off[0], com_off[1], rot, \
                                                    com_vel[0], com_vel[1], ang_vel]])
        if self.latent_feedback:
            state = np.concatenate([state, self.latent_obs])

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

        self.latent_obs = np.zeros(self.state_dim)

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
