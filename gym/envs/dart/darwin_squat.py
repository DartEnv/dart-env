__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from scipy.optimize import minimize
from pydart2.collision_result import CollisionResult
from pydart2.bodynode import BodyNode
import pydart2.pydart2_api as papi
import random
from random import randrange
import pickle
import copy, os
from gym.envs.dart.dc_motor import DCMotor

from gym.envs.dart.darwin_utils import *
from gym.envs.dart.parameter_managers import *
from gym.envs.dart.action_filter import *
import time

DEBUG = True

from pydart2.utils.transformations import euler_from_matrix, quaternion_from_matrix, euler_from_quaternion

class DartDarwinSquatEnv(dart_env.DartEnv, utils.EzPickle):
    WALK, SQUATSTAND, STEPPING, FALLING, HOP, CRAWL, STRANGEWALK, KUNGFU, BONGOBOARD, CONSTANT = list(range(10))

    def __init__(self):

        obs_dim = 40

        self.streaming_mode = False     # Mimic the streaming reading mode on the robot
        self.state_cache = [None, 0.0]
        self.gyro_cache = [None, 0.0]
        self.action_head_past = [None, None]

        self.precomputed_dynamics_parameters_set = None
        self.initial_state_set = None

        self.root_input = True
        self.include_heading = True
        self.include_base_linear_info = False
        self.terminate_sliding_contact = False
        self.foot_contact_buffer = [None, None]
        self.transition_input = False  # whether to input transition bit
        self.last_root = [0, 0]
        if self.include_heading:
            self.last_root = [0, 0, 0]
        self.fallstate_input = False

        self.adjustable_leg_compliance = False

        self.input_leg_tracking_error = False

        self.input_contact_bin = False
        self.num_contact_bin = 4

        self.input_range_sensor = False
        self.range_sensor_param = [0.1, 10]  # radius from the foot center, number of sensors

        self.variation_scheduling = None#[[0.0, {'obstacle_height_range': 0.003}], [1.0, {'obstacle_height_range': 0.003}]]
        self.assist_timeout = 0.0
        self.assist_schedule = [[0.0, [2000, 2000]], [2.0, [1500, 1500]], [4.0, [1125.0, 1125.0]]]

        self.gyro_only_mode = False
        self.leg_only_observation = True   # only read the leg motor states
        self.leg_only_action = True

        self.action_filtering = 3 # window size of filtering, 0 means no filtering
        self.action_filter_cache = []
        self.butterworth_filter = False

        self.action_delay = 0.0
        self.action_queue = []

        self.future_ref_pose = 0  # step of future trajectories as input

        self.obs_cache = []
        self.multipos_obs = 2 # give multiple steps of position info instead of pos + vel

        self.pose_dim = 12 if self.leg_only_observation else 20
        if self.multipos_obs > 0:
            obs_dim = self.pose_dim * self.multipos_obs

        self.kp_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.kd_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.use_discrete_action = False

        self.use_sysid_model = True

        if self.use_sysid_model:
            self.param_manager = darwinParamManager(self)
            #self.param_manager.activated_param.remove(self.param_manager.NEURAL_MOTOR)
        else:
            self.param_manager = darwinSquatParamManager(self)

        self.use_SPD = False

        self.use_DCMotor = False
        self.NN_motor = False
        self.NN_motor_hid_size = 5 # assume one layer
        self.NN_motor_parameters = [np.random.random((2, self.NN_motor_hid_size)), np.random.random(self.NN_motor_hid_size),
                                    np.random.random((self.NN_motor_hid_size, 2)), np.random.random(2)]
        self.NN_motor_bound = [[200.0, 1.0], [0.0, 0.0]]

        self.supress_all_randomness = False
        self.use_settled_initial_states = False
        self.limited_joint_vel = True
        self.joint_vel_limit = 20000.0
        self.train_UP = False
        self.noisy_input = False
        self.noisy_action = 0.0
        self.resample_MP = False
        self.range_robust = 0.05 # std to sample at each step
        self.randomize_timestep = False
        self.randomize_action_delay = False
        self.load_keyframe_from_file = True
        self.randomize_gravity_sch = False
        self.randomize_obstacle = False
        self.randomize_gyro_bias = False
        self.use_zeroed_gyro = True
        self.gyro_bias = np.array([0.0, 0.0])
        self.height_drop_threshold = 0.8    # terminate if com height drops for this amount
        self.orientation_threshold = 1.0    # terminate if body rotates for this amount
        self.control_interval = 0.03  # control every 50 ms
        self.sim_timestep = 0.002
        self.forward_reward = 10.0
        self.velocity_clip = 10.3
        self.contact_pen = 0.0
        self.kp = None
        self.kd = None
        self.kc = None

        self.soft_ground = False
        self.soft_foot = False
        self.task_mode = self.STEPPING
        self.side_walk = False

        if self.gyro_only_mode:
            obs_dim = 0
            self.control_interval = 0.006

        if self.use_DCMotor:
            self.motors = DCMotor(0.0107, 8.3, 12, 193)

        obs_dim += self.future_ref_pose * self.pose_dim

        if self.root_input:
            obs_dim += 4
            if self.include_heading:
                obs_dim += 2

        if self.include_base_linear_info:
            obs_dim += 3

        if self.fallstate_input:
            obs_dim += 2

        if self.transition_input:
            obs_dim += 1

        if self.train_UP:
            obs_dim += self.param_manager.param_dim

        if self.task_mode == self.BONGOBOARD: # add observation about board
            obs_dim += 3

        if self.input_leg_tracking_error:
            obs_dim += 12

        if self.input_contact_bin:
            obs_dim += 2 * self.num_contact_bin

        if self.input_range_sensor:
            obs_dim += 2 * self.range_sensor_param[1]

        self.act_dim = 20
        if self.leg_only_action:
            self.act_dim = 12

        if self.adjustable_leg_compliance:
            obs_dim += 2
            self.act_dim += 2  # one for each ankle
        self.control_bounds = np.array([-np.ones(self.act_dim, ), np.ones(self.act_dim, )])

        self.observation_buffer = []
        self.obs_delay = 0

        self.gravity_sch = [[0.0, np.array([0, 0, -9.81])]]

        self.initialize_falling = False # initialize darwin to have large ang vel so that it falls


        self.delta_angle_scale = 0.3

        self.alive_bonus = 5.0
        self.energy_weight = 0.001
        self.work_weight = 0.005
        self.pose_weight = 5.0
        self.upright_weight = 0.0
        self.comvel_pen = 0.0
        self.compos_pen = 0.0
        self.compos_range = 0.5

        self.cur_step = 0

        self.torqueLimits = 10.0

        self.t = 0
        self.target = np.zeros(20, )
        self.tau = np.zeros(20, )

        self.include_obs_history = 1
        self.include_act_history = 3
        self.input_obs_difference = True
        self.input_difference_sign = False    # Whether to use the sign instead of actual value for joint velocity
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history
        if self.leg_only_observation:
            obs_perm_base = np.array(
                [-6, -7, -8, -9, -10, -11, -0.0001, -1, -2, -3, -4, -5,
                 -18, -19, -20, -21, -22, -23, -12, -13, -14, -15, -16, -17])
        else:
            obs_perm_base = np.array(
                [-3, -4, -5, -0.0001, -1, -2, -6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13,
                 -23, -24, -25, -20, -21, -22, -26, 27, -34, -35, -36, -37, -38, -39, -28, -29, -30, -31, -32, -33])
        if self.gyro_only_mode:
            obs_perm_base = np.array([])
        if self.leg_only_action:
            act_perm_base = np.array(
                [-6, -7, -8, -9, -10, -11, -0.001, -1, -2, -3, -4, -5])
        else:
            act_perm_base = np.array(
                [-3, -4, -5, -0.0001, -1, -2, -6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13])
        if self.adjustable_leg_compliance:
            act_perm_base = np.concatenate([act_perm_base, [len(act_perm_base)+1, len(act_perm_base)]])
        self.obs_perm = np.copy(obs_perm_base)

        if self.root_input:
            beginid = len(obs_perm_base)
            if self.include_heading:
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid-0.0001, beginid + 1, -beginid - 2,  -beginid - 3, beginid + 4, -beginid - 5]])
            else:
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid-0.0001, beginid + 1, -beginid - 2, beginid + 3]])
        if self.include_base_linear_info:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [beginid, - beginid - 1, beginid + 2]])
        if self.fallstate_input:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid-0.0001, beginid + 1]])
        if self.transition_input:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [beginid]])
        if self.task_mode == self.BONGOBOARD:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid-0.0001, beginid + 1, -beginid - 2]])

        if self.input_leg_tracking_error:
            # Input the tracking errors for the leg dofs
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid-6, -beginid -7, -beginid - 8,
                   -beginid-9, -beginid -10, -beginid - 11, -beginid, -beginid-1, -beginid-2,
                   -beginid-3, -beginid-4, -beginid-5]])

        if self.input_contact_bin:
            beginid = len(obs_perm_base)
            contact_bin_perm = []
            for i in range(self.num_contact_bin):
                contact_bin_perm.append(beginid+i+self.num_contact_bin)
            for i in range(self.num_contact_bin):
                contact_bin_perm.append(beginid+i)
            obs_perm_base = np.concatenate([obs_perm_base, contact_bin_perm])

        if self.input_range_sensor:
            beginid = len(obs_perm_base)
            range_sensor_perm = []
            for i in range(self.range_sensor_param[1]):
                range_sensor_perm.append(beginid + i + self.range_sensor_param[1])
            for i in range(self.range_sensor_param[1]):
                range_sensor_perm.append(beginid + i)
            obs_perm_base = np.concatenate([obs_perm_base, range_sensor_perm])

        if self.adjustable_leg_compliance:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [beginid + 1, beginid]])

        if self.train_UP:
            obs_perm_base = np.concatenate([obs_perm_base, np.arange(len(obs_perm_base), len(obs_perm_base) + len(
                self.param_manager.activated_param))])

        self.obs_perm = np.copy(obs_perm_base)

        for i in range(self.include_obs_history - 1):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(obs_perm_base) * (np.abs(obs_perm_base) + len(self.obs_perm))])
        for i in range(self.include_act_history):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(act_perm_base) * (np.abs(act_perm_base) + len(self.obs_perm))])
        self.act_perm = np.copy(act_perm_base)

        if self.use_discrete_action:
            from gym import spaces
            disc_nvec = [3] * len(self.control_bounds[0])
            if self.adjustable_leg_compliance:
                disc_nvec += [2] * 2
            self.action_space = spaces.MultiDiscrete(disc_nvec)
            #self.action_space = spaces.MultiDiscrete([5,5,5,5,5,5, 3,3, 7,7,7,7,7,7, 7,7,7,7,7,7])

        model_file_list = ['darwinmodel/ground1.urdf', 'darwinmodel/darwin_nocollision.URDF', 'darwinmodel/coord.urdf', 'darwinmodel/tracking_box.urdf', 'darwinmodel/bongo_board.urdf', 'darwinmodel/robotis_op2.urdf']
        if self.soft_foot:
            model_file_list[5] = 'darwinmodel/robotis_op2_softfoot.urdf'
        if self.task_mode != self.BONGOBOARD:
            model_file_list.remove('darwinmodel/bongo_board.urdf')
        if self.soft_ground:
            model_file_list[0] = 'darwinmodel/soft_ground.skel'
            model_file_list.insert(4, 'darwinmodel/soft_ground.urdf')

        dart_env.DartEnv.__init__(self, model_file_list, int(self.control_interval / self.sim_timestep), obs_dim,
                                  self.control_bounds, dt=self.sim_timestep, disableViewer=True, action_type="continuous" if not self.use_discrete_action else "discrete")

        self.body_parts = [bn for bn in self.robot_skeleton.bodynodes if 'SHOE' not in bn.name and 'base_link' not in bn.name]
        self.body_part_ids = np.array([bn.id for bn in self.body_parts])

        self.actuated_dofs = [df for df in self.robot_skeleton.dofs if 'root' not in df.name and 'shoe' not in df.name]
        self.actuated_dof_ids = [df.id for df in self.actuated_dofs]
        self.observed_dof_ids = [df.id for df in self.robot_skeleton.dofs if 'root' not in df.name and 'shoe' not in df.name]

        self.obstacle_bodynodes = [self.dart_world.skeletons[0].bodynode(bn) for bn in sorted([b.name for b in self.dart_world.skeletons[0].bodynodes if 'obstacle' in b.name])]

        if self.leg_only_observation:
            self.observed_dof_ids = [df.id for df in self.robot_skeleton.dofs if 'root' not in df.name and 'shoe' not in df.name
                                     and 'shoulder' not in df.name and 'arm' not in df.name
                                     and 'pan' not in df.name and 'tilt' not in df.name]

        self.mass_ratios = np.ones(len(self.body_part_ids))
        self.inertia_ratios = np.ones(len(self.body_part_ids))

        self.left_foot_shoe_bodies = [bn for bn in self.robot_skeleton.bodynodes if
                                      'SHOE_PIECE' in bn.name and '_L' in bn.name]
        self.right_foot_shoe_bodies = [bn for bn in self.robot_skeleton.bodynodes if
                                       'SHOE_PIECE' in bn.name and '_R' in bn.name]
        self.left_foot_shoe_ids = [bn.id for bn in self.left_foot_shoe_bodies]
        self.right_foot_shoe_ids = [bn.id for bn in self.right_foot_shoe_bodies]
        self.left_shoe_dofs = [df for df in self.robot_skeleton.dofs if 'shoe' in df.name and '_l' in df.name]
        self.right_shoe_dofs = [df for df in self.robot_skeleton.dofs if 'shoe' in df.name and '_r' in df.name]
        if self.soft_foot:
            for df in self.left_shoe_dofs:
                df.set_spring_stiffness(3000)
                df.set_damping_coefficient(5)
            for df in self.right_shoe_dofs:
                df.set_spring_stiffness(3000)
                df.set_damping_coefficient(5)

        # crawl
        if self.task_mode == self.CRAWL:
            self.permitted_contact_ids = self.body_part_ids[[-1, -2, -7, -8, 5, 10]]  # [-1, -2, -7, -8]
            self.init_root_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.08, 0.0])
        else:
            # normal pose
            self.permitted_contact_ids = self.body_part_ids[[-1, -2, -7, -8, 5, 10]]
            self.init_root_pert = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])

        # For tacking foot sliding
        self.left_foot_id = self.robot_skeleton.bodynodes[-1].id
        self.right_foot_id = self.robot_skeleton.bodynodes[-7].id

        if self.side_walk:
            self.init_root_pert = np.array([0.0, 0., -1.57, 0.0, 0.0, 0.0])

        if len(self.left_foot_shoe_ids) > 0:
            self.permitted_contact_ids = np.concatenate([self.permitted_contact_ids, self.left_foot_shoe_ids,
                                                         self.right_foot_shoe_ids])


        self.orig_bodynode_masses = [bn.mass() for bn in self.body_parts]
        self.orig_bodynode_inertias = [bn.I for bn in self.body_parts]

        self.dart_world.set_gravity([0, 0, -9.81])

        self.dupSkel = self.dart_world.skeletons[1]
        self.dupSkel.set_mobile(False)
        self.dart_world.skeletons[2].set_mobile(False)

        if self.soft_ground:
            self.dart_world.set_collision_detector(1)
        elif self.task_mode == self.BONGOBOARD:
            self.dart_world.set_collision_detector(3)
        else:
            self.dart_world.set_collision_detector(2)

        self.robot_skeleton.set_self_collision_check(True)


        collision_filter = self.dart_world.create_collision_filter()
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_PELVIS_L'),
                                           self.robot_skeleton.bodynode('MP_THIGH2_L'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_PELVIS_R'),
                                           self.robot_skeleton.bodynode('MP_THIGH2_R'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_ARM_HIGH_L'),
                                           self.robot_skeleton.bodynode('l_hand'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_ARM_HIGH_R'),
                                           self.robot_skeleton.bodynode('r_hand'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_TIBIA_R'),
                                           self.robot_skeleton.bodynode('MP_ANKLE2_R'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_TIBIA_L'),
                                           self.robot_skeleton.bodynode('MP_ANKLE2_L'))

        for i in range(len(self.left_foot_shoe_bodies)):
            for j in range(i+1, len(self.left_foot_shoe_bodies)):
                collision_filter.add_to_black_list(self.left_foot_shoe_bodies[i], self.left_foot_shoe_bodies[j])
        for i in range(len(self.right_foot_shoe_bodies)):
            for j in range(i+1, len(self.right_foot_shoe_bodies)):
                collision_filter.add_to_black_list(self.right_foot_shoe_bodies[i], self.right_foot_shoe_bodies[j])


        self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(1.0)
        for bn in self.robot_skeleton.bodynodes:
            bn.set_friction_coeff(1.0)
        self.robot_skeleton.bodynode('l_hand').set_friction_coeff(2.0)
        self.robot_skeleton.bodynode('r_hand').set_friction_coeff(2.0)

        self.add_perturbation = False
        self.perturbation_parameters = [0.5, 0.2, 1.5, [0.5, 1.5], 1]  # begin time, duration, interval, magnitude, bodyid

        for j in self.actuated_dofs:
            j.set_damping_coefficient(0.515)
            j.set_coulomb_friction(0.0)

        self.init_position_x = self.robot_skeleton.bodynode('MP_BODY').C[0]

        # set joint limits according to the measured one
        for i in range(len(self.actuated_dofs)):
            self.actuated_dofs[i].set_position_lower_limit(JOINT_LOW_BOUND[i] - 0.01)
            self.actuated_dofs[i].set_position_upper_limit(JOINT_UP_BOUND[i] + 0.01)

        self.permitted_contact_bodies = [self.robot_skeleton.bodynodes[id] for id in self.permitted_contact_ids]

        if self.task_mode == self.BONGOBOARD:
            self.permitted_contact_bodies += [b for b in self.dart_world.skeletons[4].bodynodes]

        self.initial_local_coms = [b.local_com() for b in self.body_parts]

        ################# temp code, ugly for now, should fix later ###################################
        if self.use_sysid_model:
            # self.param_manager.controllable_param = [self.param_manager.NEURAL_MOTOR,
            #                                          self.param_manager.GROUP_JOINT_DAMPING,
            #                                          self.param_manager.TORQUE_LIM, self.param_manager.COM_OFFSET,
            #                                          self.param_manager.GROUND_FRICTION,
            #                                          self.param_manager.KP_RATIO_NEW, self.param_manager.KD_RATIO_NEW]
            # self.param_manager.set_simulator_parameters(
            #     np.array([0.22600929, 0.10767238, 0.55781467, 0.17783977, 0.25233775,
            #                0.43583618, 0.00407056, 0.25220361, 0.86859472, 0.47251642,
            #                0.79428188, 0.5490745 , 0.29529473, 0.84082059, 0.99960931,
            #                0.16142782, 0.78019638, 0.00248998, 0.97988873, 0.89419269,
            #                0.32932616, 0.06995249, 0.31929577, 0.00185668, 0.32337695,
            #                0.00757815, 0.77566245, 0.4636164 , 0.61485352, 0.98097431,
            #                0.52242523, 0.53870455, 0.03200784, 0.39488985, 0.83295163,
            #                0.39685312, 0.29405877, 0.52418615, 0.87898638, 0.40396424,
            #                0.21688968, 0.4615679 , 0.53865347, 0.16924662, 0.0122813 ,
            #                0.56959068, 0.62130358, 0.32067348, 0.535078  , 0.52364651]))
            # self.param_manager.controllable_param.remove(self.param_manager.NEURAL_MOTOR)
            # self.param_manager.set_bounds(np.array([0.70163534, 0.77084022, 0.99989368, 0.79073859, 0.7007047 ,
            #                                            0.31475583, 0.54099058, 0.99993862, 0.64150573, 0.52056636,
            #                                            0.79718917, 0.99999997, 0.60404622, 0.37680236, 0.9769624 ,
            #                                            0.79751671, 0.35583425, 0.21579834, 0.6928588 , 0.81740399,
            #                                            0.52959217, 0.78898541, 0.69143631]),
            #                               np.array([2.83731235e-01, 4.15493724e-01, 7.92033447e-01, 3.57466110e-01,
            #                                            3.13241712e-01, 9.07380889e-06, 2.49374521e-01, 6.24070954e-01,
            #                                            2.03030731e-01, 2.09808424e-01, 3.21225840e-01, 7.09669298e-01,
            #                                            1.88410694e-01, 7.26213373e-02, 1.46799396e-01, 3.89002425e-01,
            #                                            8.69687012e-04, 1.97590897e-07, 3.73029138e-01, 2.64782073e-01,
            #                                            3.53021367e-05, 1.76376197e-01, 3.48950533e-01]))

            self.param_manager.set_simulator_parameters(
                np.array([2.87159059e-01, 4.03160514e-01, 4.36576586e-01, 3.86221239e-01,
                          7.85789054e-01, 1.04277029e-01, 3.64862787e-01, 3.98563863e-01,
                          9.36966648e-01, 9.56131312e-01, 8.74345365e-01, 8.39548565e-01,
                          9.90829332e-01, 1.07563860e-01, 6.43309153e-01, 9.88438984e-01,
                          2.85672012e-01, 9.67511394e-01, 5.98024447e-01, 1.59794372e-01,
                          9.97536608e-01, 4.88691407e-01, 5.01293655e-01, 7.95171350e-01,
                          9.95825152e-02, 7.09580629e-03, 4.66536839e-01, 5.25860303e-01,
                          8.20514312e-01, 9.35216575e-04, 2.74604822e-01, 7.11505683e-02,
                          4.56312986e-01, 9.28976189e-01, 7.45092860e-01, 5.09716306e-01,
                          6.45103472e-01, 7.33841140e-01, 3.06389080e-01, 9.99043259e-01,
                          2.37641857e-01]))
            self.param_manager.controllable_param.remove(self.param_manager.NEURAL_MOTOR)
            self.param_manager.set_bounds(np.array([0.39913768, 0.51181744, 0.71311956, 0.52887732, 0.88725472,
                                                    0.25034288, 0.45227712, 0.56491894, 0.96844332, 0.97803881,
                                                    0.8564158, 0.43046669, 1.0, 0.6]),
                                          np.array([0.21157757, 0.27855955, 0.3342329, 0.33561788, 0.65942584,
                                                    0.05213874, 0.19820029, 0.27135438, 0.72653479, 0.87677413,
                                                    0.36699146, 0.20532608, 0.0, 0.1]))

            # self.param_manager.set_bounds(np.array([0.62754478, 1., 1., 0.91796176, 0.99481419,
            #                                         0.62411285, 0.58039399, 1., 1., 1.,
            #                                         1., 0.51500521, 1., 0.6]),
            #                               np.array([0.1620302, 0.23465617, 0.18431452, 0.24797362, 0.53741423,
            #                                         0., 0.052823, 0.22073834, 0.34243664, 0.74839466,
            #                                         0., 0., 0., 0.1]))
            # 1 std for setting the range
            # self.param_manager.set_bounds(np.array([0.49049991, 0.62215316, 0.84114612, 0.59905219, 0.98277157,
            #                                         0.32097985, 0.54746922, 0.58546215, 1., 1.,
            #                                         1., 0.48281712, 1., 0.6]),
            #                               np.array([0.14208371, 0.23364703, 0.15188524, 0.23584708, 0.56458775,
            #                                         0., 0.22041629, 0.1860213, 0.59190119, 0.85595963,
            #                                         0.3238342, 0.13586187, 0.7084698, 0.07304083]))


        self.default_kp_ratios = np.copy(self.kp_ratios)
        self.default_kd_ratios = np.copy(self.kd_ratios)
        ######################################################################################################

        if self.use_SPD:
            self.Kp = np.diagflat([0.0] * 6 + [500.0] * (self.robot_skeleton.ndofs - 6))
            self.Kd = np.diagflat([0.0] * 6 + [1.0] * (self.robot_skeleton.ndofs - 6))

        print('Total mass: ', self.robot_skeleton.mass())
        print('Bodynodes: ', [b.name for b in self.robot_skeleton.bodynodes])

        if self.task_mode == self.WALK:
            self.setup_walk()
        elif self.task_mode == self.STEPPING:
            self.setup_stepping()
        elif self.task_mode == self.SQUATSTAND:
            self.setup_squatstand()
        elif self.task_mode == self.FALLING:
            self.setup_fall()
        elif self.task_mode == self.HOP:
            self.setup_hop()
        elif self.task_mode == self.CRAWL:
            self.setup_crawl()
        elif self.task_mode == self.STRANGEWALK:
            self.setup_strangewalk()
        elif self.task_mode == self.KUNGFU:
            self.setup_kungfu()
        elif self.task_mode == self.BONGOBOARD:
            self.setup_bongoboard()
        elif self.task_mode == self.CONSTANT:
            self.setup_constref()

        if self.butterworth_filter:
            self.action_filter = ActionFilter(self.act_dim, 3, int(1.0/self.dt), low_cutoff=0.0, high_cutoff=10.0)

        # self.set_robot_optimization_parameters(np.array([-0.27573608, -0.04001381,  0.16576692, -0.45604828,  0.83507119,
        # 0.2363036 , -0.37442629, -0.77073466,  0.69862929, -0.85059406]) * 0.01)

        self.init_alive_bonus = self.alive_bonus
        self.init_x_progress = 0.0

        utils.EzPickle.__init__(self)


    def setup_walk(self): # step up walk task
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe.txt')
            rig_keyframe = np.loadtxt(fullpath)

            '''for i in range(len(rig_keyframe)):
                rig_keyframe[i][0:6] = 0.0
                rig_keyframe[i][1] = -0.5
                rig_keyframe[i][2] = 0.75
                rig_keyframe[i][4] = 0.5
                rig_keyframe[i][5] = -0.75'''

            self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.25
            for i in range(20):
                for k in range(1, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.25
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        self.compos_range = 0.5
        self.forward_reward = 10.0
        self.delta_angle_scale = 0.3
        self.init_root_pert = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/stand_init.txt'))

    def setup_stepping(self): # step up stepping task
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe_step.txt')
            rig_keyframe = np.loadtxt(fullpath)

            self.interp_sch = [[0.0, rig_keyframe[0]]]
            #rig_keyframe = [HW2SIM_INDEX(v) for v in VAL2RADIAN(rig_keyframe)]
            #self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.2
            for i in range(3):
                for k in range(0, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.03
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        self.compos_range = 0.5
        self.forward_reward = 0.0
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/halfsquat_init.txt'))

    def setup_constref(self): # constant reference motion
        const_pose = VAL2RADIAN(0.5 * (np.array([2450, 2250, 1714, 1646, 1750, 2376,
                                                                 2047, 2171,
                                                                 2032, 2039, 2795, 648, 1241, 2040, 2041, 2060, 1281,
                                                                 3448, 2855, 2073]) + np.array(
                [2450, 2250, 1714, 1646, 1750, 2376,
                 2048, 2048,
                 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])))
        self.interp_sch = [[0.0, const_pose], [30.0, const_pose]]
        self.compos_range = 0.5
        self.forward_reward = 20.0
        self.delta_angle_scale = 0.6
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(
                os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/halfsquat_init.txt'))

    def setup_squatstand(self): # set up squat stand task
        self.interp_sch = [[0.0, pose_stand_rad],
                           [2.0, pose_squat_rad],
                           [3.5, pose_squat_rad],
                           [4.0, pose_stand_rad],
                           [5.0, pose_stand_rad],
                           [7.0, pose_squat_rad],
                           ]
        self.compos_range = 100.0
        self.forward_reward = 0.0
        self.init_root_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.delta_angle_scale = 0.2
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt('darwinmodel/stand_init.txt')

    def setup_fall(self): # set up the falling task
        self.interp_sch = [[0.0, 0.5 * (pose_squat_rad + pose_stand_rad)],
                           [4.0, 0.5 * (pose_squat_rad + pose_stand_rad)]]
        self.compos_range = 100.0
        self.forward_reward = 0.0
        self.contact_pen = 0.05
        self.delta_angle_scale = 0.6
        self.alive_bonus = 8.0
        self.height_drop_threshold = 10.0
        self.orientation_threshold = 10.0
        self.initialize_falling = True
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/stand_init.txt'))

    def setup_hop(self): # set up hop task
        p0 = 0.2 * pose_stand_rad + 0.8 * pose_squat_rad
        p1 = 0.85 * pose_stand_rad + 0.15 * pose_squat_rad
        p1[0] -= 0.7
        p1[3] += 0.7
        self.interp_sch = []
        curtime = 0
        for i in range(20):
            self.interp_sch.append([curtime, p0])
            self.interp_sch.append([curtime+0.2, p1])
            #self.interp_sch.append([curtime+0.4, p0])
            curtime += 0.4

        self.compos_range = 100.0
        self.forward_reward = 10.0
        self.init_root_pert = np.array([0.0, 0.16, 0.0, 0.0, 0.0, 0.0])
        self.delta_angle_scale = 0.3
        self.energy_weight = 0.005
        if self.use_settled_initial_states:
            self.init_states_candidates = np.loadtxt(os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/squat_init.txt'))

    def setup_crawl(self): # set up crawling task
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe_crawl.txt')
            rig_keyframe = np.loadtxt(fullpath)

            self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.15
            for i in range(10):
                for k in range(0, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.15
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        self.compos_range = 0.3
        self.forward_reward = 10.0
        self.height_drop_threshold = 10.0
        self.orientation_threshold = 10.0

    def setup_strangewalk(self):
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe_strangewalk.txt')
            rig_keyframe = np.loadtxt(fullpath)

            self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.25
            for i in range(20):
                for k in range(0, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.25
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        self.compos_range = 0.3
        self.delta_angle_scale = 0.2
        self.init_root_pert = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
        self.forward_reward = 10.0
        self.upright_weight = 1.0

    def setup_kungfu(self):
        self.load_keyframe_from_file = True
        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe_kungfu.txt')
            rig_keyframe = np.loadtxt(fullpath)

            self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.3
            for i in range(1):
                for k in range(0, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.1
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        self.compos_range = 0.3
        self.forward_reward = 0.0
        self.upright_weight = 1.0

    def setup_bongoboard(self):
        from pydart2.constraints import BallJointConstraint
        p = 0.5 * (pose_squat_rad + pose_stand_rad)
        p[1] = -0.5
        p[2] = 0.75
        p[4] = 0.5
        p[5] = -0.75
        p[9] -= 0.3
        p[15] += 0.3
        p[13] -= 0.3
        p[19] += 0.3
        p = np.clip(p, JOINT_LOW_BOUND, JOINT_UP_BOUND)
        self.interp_sch = [[0, p], [5, p]]

        self.compos_range = 0.3
        self.forward_reward = 0.0
        self.init_root_pert = np.array([0.0, 0.06, 0.0, 0.0, 0.0, 0.0])
        self.delta_angle_scale = 1.0
        self.upright_weight = 0.5
        self.comvel_pen = 0.5
        self.compos_pen = 1.0
        if self.task_mode == self.BONGOBOARD:
            self.dart_world.skeletons[4].bodynodes[0].set_friction_coeff(20.0)
        #self.dart_world.skeletons[0].bodynodes[2].shapenodes[0].set_offset([0, 0.5, 0])
        #self.dart_world.skeletons[0].bodynodes[2].shapenodes[1].set_offset([0, 0.5, 0])
        self.param_manager.MU_UP_BOUNDS[self.param_manager.GROUND_FRICTION] = [2.0]
        self.param_manager.MU_LOW_BOUNDS[self.param_manager.GROUND_FRICTION] = [1.0]




    def adjust_root(self): # adjust root dof such that foot is roughly flat
        q = self.robot_skeleton.q
        q[1] -= -1.57 - np.array(euler_from_matrix(self.robot_skeleton.bodynode('MP_ANKLE2_L').T[0:3, 0:3], 'sxyz'))[1]

        if not self.soft_foot:
            q[5] += -0.335 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2]])
        else:
            q[5] += -0.335 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2], self.left_foot_shoe_bodies[0].C[2], self.right_foot_shoe_bodies[0].C[2]])
        self.robot_skeleton.q = q

    def get_body_quaternion(self):
        q = quaternion_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T)
        return q

    def get_sim_bno55(self, supress_randomization=False):
        # simulate bno55 reading
        tinv = np.linalg.inv(np.array(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3]))
        angvel = self.robot_skeleton.bodynode('MP_BODY').com_spatial_velocity()[0:3]
        langvel = np.dot(tinv, angvel)
        euler = np.array(euler_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3], 'sxyz'))

        if self.randomize_gyro_bias and not supress_randomization and not self.supress_all_randomness:
            euler[0:2] += self.gyro_bias
            # add noise
            euler += np.random.uniform(-0.01, 0.01, 3)
            langvel += np.random.uniform(-0.1, 0.1, 3)
        # if euler[0] > 1.5:
        #     euler[0] -= np.pi
        return np.array([euler[0], euler[1], euler[2], langvel[0], langvel[1], langvel[2]])
        # return np.array(
        #     [euler[0] + 0.08040237422714677, euler[1] - 0.075 - 0.12483721034195938, euler[2], langvel[0], langvel[1],
        #      langvel[2]])

    def falling_state(self): # detect if it's falling fwd/bwd or left/right
        gyro = self.get_sim_bno55()
        fall_flags = [0, 0]
        if np.abs(gyro[0]) > 0.5:
            fall_flags[0] = np.sign(gyro[0])
        if np.abs(gyro[1]) > 0.5:
            fall_flags[1] = np.sign(gyro[1])
        return fall_flags

    def spd(self, target_q):
        invM = np.linalg.inv(self.robot_skeleton.M + self.Kd * self.sim_dt)
        p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.sim_dt - np.concatenate([[0.0]*6, target_q]))
        d = -self.Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.sim_dt
        return tau[6:]

    def advance(self, a):
        if self._get_viewer() is not None:
            if hasattr(self._get_viewer(), 'key_being_pressed'):
                if self._get_viewer().key_being_pressed is not None:
                    if self._get_viewer().key_being_pressed == b'p':
                        self.paused = not self.paused
                        time.sleep(0.1)

        if self.paused and self.t > 0:
            return
        clamped_control = np.array(a)

        # if self.butterworth_filter:
        #     clamped_control = self.action_filter.filter_action(clamped_control)
        self.action_buffer.append(np.copy(clamped_control))

        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if clamped_control[i] < self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]

        if self.adjustable_leg_compliance:
            self.left_leg_compliance = clamped_control[-2]
            self.right_leg_compliance = clamped_control[-1]
            clamped_control = clamped_control[0:-2]
            if self.left_leg_compliance > 0.0:
                self.robot_skeleton.bodynodes[5].shapenodes[0].set_visual_aspect_rgba([0.5, 0.5, 1.0, 1.0])
            else:
                self.robot_skeleton.bodynodes[5].shapenodes[0].set_visual_aspect_rgba([1.0, 0.0, 0.0, 1.0])

            if self.right_leg_compliance > 0.0:
                self.robot_skeleton.bodynodes[10].shapenodes[0].set_visual_aspect_rgba([0.5, 0.5, 1.0, 1.0])
            else:
                self.robot_skeleton.bodynodes[10].shapenodes[0].set_visual_aspect_rgba([1.0, 0.0, 0.0, 1.0])

        self.ref_target = self.get_ref_pose(self.t)

        clamped_control += np.random.uniform(-self.noisy_action, self.noisy_action, len(clamped_control))

        if self.leg_only_action:
            clamped_control = np.concatenate([np.zeros(8), clamped_control])

        # filter delta signal
        if self.butterworth_filter:
            if self.leg_only_action:
                clamped_control[8:] = self.action_filter.filter_action(clamped_control[8:])
            else:
                clamped_control = self.action_filter.filter_action(clamped_control)
        cur_target = self.ref_target + clamped_control * self.delta_angle_scale

        # MMHACK
        # randomly reset the filter cache
        # if np.random.random() < 0.01:
        #     self.reset_filtering()

        # filter absolute signal
        # if self.butterworth_filter:
        #     if self.leg_only_action:
        #         cur_target[8:] = self.action_filter.filter_action(cur_target[8:])
        #     else:
        #         cur_target = self.action_filter.filter_action(cur_target)

        self.action_filter_cache.append(cur_target)
        if len(self.action_filter_cache) > self.action_filtering:
            self.action_filter_cache.pop(0)
        if self.action_filtering > 0:
            cur_target = np.mean(self.action_filter_cache, axis=0)

        cur_target = np.clip(cur_target, CONTROL_LOW_BOUND, CONTROL_UP_BOUND)

        self.apply_target_pose(cur_target)
        self.target_pose_history.append(cur_target)

    def _bodynode_spd(self, bn, kp, dof, target_vel=None):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_vel is not None:
            self.Kd = self.Kp
            self.Kp *= 0
        invM = 1.0 / (bn.mass() + self.Kd * self.sim_dt)
        p = -self.Kp * (bn.C[dof] + bn.dC[dof] * self.sim_dt)
        if target_vel is None:
            target_vel = 0.0
        d = -self.Kd * (bn.dC[dof] - target_vel)
        qddot = invM * (-bn.C[dof] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt
        return tau

    def apply_target_pose(self, target_pose):
        dup_pos = np.concatenate([[0.0] * 6, target_pose])
        dup_pos[4] = 0.5
        self.dupSkel.set_positions(dup_pos)
        self.dupSkel.set_velocities(np.zeros(len(target_pose) + 6))

        self.action_queue.append([target_pose, self.t + self.action_delay])
        if self.add_perturbation and self.t >= self.perturbation_parameters[0]:# and not self.supress_all_randomness:
            if (self.t - self.perturbation_parameters[0]) % self.perturbation_parameters[2] <= 0.04:
                force_mag = np.random.uniform(self.perturbation_parameters[3][0], self.perturbation_parameters[3][1])
                force_dir = np.array([np.random.uniform(-1, 1), np.random.uniform(-0.3, 0.3), np.random.uniform(-0.1, 0.1)])
                self.perturb_force = force_dir / np.linalg.norm(force_dir) * force_mag
            elif (self.t - self.perturbation_parameters[0]) % self.perturbation_parameters[2] > \
                    self.perturbation_parameters[1]:
                self.perturb_force *= 0

        if self.streaming_mode:
            if self.action_head_past[1] is None:
                self.action_head_past[0] = np.copy(target_pose)
                self.action_head_past[1] = np.copy(target_pose)

        for i in range(self.frame_skip):
            if self.streaming_mode:
                if self.t + i * self.sim_dt - self.state_cache[1] > self.state_cache[2]:
                    self.action_head_past[1] = np.copy(self.action_head_past[0])
                    self.action_head_past[0] = np.copy(target_pose)
                self.target = np.copy(self.action_head_past[1])
            else:
                if len(self.action_queue) > 0 and self.t + i * self.sim_dt >= self.action_queue[0][1]:
                    self.target = np.copy(self.action_queue[0][0])
                    self.action_queue.pop(0)

            if self.use_SPD:
                self.tau = self.spd(target_pose)
            else:
                self.tau = self.PID()

            if self.add_perturbation:
                self.robot_skeleton.bodynodes[self.perturbation_parameters[4]].add_ext_force(self.perturb_force)
            # if self.t < 0.5:
            #     self.robot_skeleton.bodynode('MP_BODY').add_ext_force([3.0,0,0])

            if self.t < self.assist_timeout:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_BODY'), self.current_pd, 1)
                self.robot_skeleton.bodynode('MP_BODY').add_ext_force(np.array([0, force, 0]))

            robot_force = np.zeros(self.robot_skeleton.ndofs)
            robot_force[self.actuated_dof_ids] = self.tau
            self.robot_skeleton.set_forces(robot_force)

            self.dart_world.step()

            if self.streaming_mode:
                if self.t + i * self.sim_dt - self.state_cache[1] > self.state_cache[2]:
                    self.state_cache = [self.robot_skeleton.q[6:], self.t + i * self.sim_dt, 0.015 + np.random.normal(0.0, 0.005)]
                if self.t + i * self.sim_dt - self.gyro_cache[1] > self.gyro_cache[2]:
                    self.gyro_cache = [self.get_sim_bno55(), self.t + i * self.sim_dt, 0.015 + np.random.normal(0.0, 0.005)]

        if not self.paused or self.t == 0:
            self.t += self.dt * 1.0
            self.cur_step += 1

    def NN_forward(self, input):
        NN_out = np.dot(np.tanh(np.dot(input, self.NN_motor_parameters[0]) + self.NN_motor_parameters[1]),
                        self.NN_motor_parameters[2]) + self.NN_motor_parameters[3]

        NN_out = np.exp(-np.logaddexp(0, -NN_out))
        return NN_out

    def PID(self):
        # print("#########################################################################3")

        if self.use_DCMotor:
            if self.kp is not None:
                kp = self.kp
                kd = self.kd
            else:
                kp = np.array([4]*20)
                kd = np.array([0.032]*20)
            pwm_command = -1 * kp * (np.array(self.robot_skeleton.q)[self.actuated_dof_ids] - self.target) - kd * np.array(self.robot_skeleton.dq)[self.actuated_dof_ids]
            tau = self.motors.get_torque(pwm_command, np.array(self.robot_skeleton.dq)[self.actuated_dof_ids])
        elif self.NN_motor:
            q = np.array(self.robot_skeleton.q)
            qdot = np.array(self.robot_skeleton.dq)
            tau = np.zeros(20, )

            input = np.vstack([np.abs(q[self.actuated_dof_ids] - self.target) * 5.0, np.abs(qdot[self.actuated_dof_ids])]).T

            NN_out = self.NN_forward(input)

            kp = NN_out[:, 0] * (self.NN_motor_bound[0][0] - self.NN_motor_bound[1][0]) + self.NN_motor_bound[1][0]
            kd = NN_out[:, 1] * (self.NN_motor_bound[0][1] - self.NN_motor_bound[1][1]) + self.NN_motor_bound[1][1]

            if len(self.kp_ratios) == 5:
                kp[0:6] *= self.kp_ratios[0]
                kp[6:8] *= self.kp_ratios[1]
                kp[8:11] *= self.kp_ratios[2]
                kp[14:17] *= self.kp_ratios[2]
                kp[11] *= self.kp_ratios[3]
                kp[17] *= self.kp_ratios[3]
                kp[12:14] *= self.kp_ratios[4]
                kp[18:20] *= self.kp_ratios[4]
                if self.adjustable_leg_compliance:
                    if self.left_leg_compliance < 0.0:
                        kp[12:14] *= 0.1
                    if self.right_leg_compliance < 0.0:
                        kp[18:20] *= 0.1
                # kp[2] = 0.0
                # kp[7] = 0.0

            if len(self.kp_ratios) == 10:
                kp[[0, 1, 2, 6, 8,9,10,11,12,13]] *= self.kp_ratios
                kp[[3, 4, 5, 7, 14,15,16,17,18,19]] *= self.kp_ratios

            if len(self.kp_ratios) == 7:
                kp[0:8] *= self.kp_ratios[0]
                kp[8] *= self.kp_ratios[1]
                kp[9] *= self.kp_ratios[2]
                kp[10] *= self.kp_ratios[3]
                kp[11] *= self.kp_ratios[4]
                kp[12] *= self.kp_ratios[5]
                kp[13] *= self.kp_ratios[6]
                kp[14] *= self.kp_ratios[1]
                kp[15] *= self.kp_ratios[2]
                kp[16] *= self.kp_ratios[3]
                kp[17] *= self.kp_ratios[4]
                kp[18] *= self.kp_ratios[5]
                kp[19] *= self.kp_ratios[6]

            if len(self.kd_ratios) == 5:
                kd[0:6] *= self.kd_ratios[0]
                kd[6:8] *= self.kd_ratios[1]
                kd[8:11] *= self.kd_ratios[2]
                kd[14:17] *= self.kd_ratios[2]
                kd[11] *= self.kd_ratios[3]
                kd[17] *= self.kd_ratios[3]
                kd[12:14] *= self.kd_ratios[4]
                kd[18:20] *= self.kd_ratios[4]
                # kd[2] = 0.0

            if len(self.kd_ratios) == 10:
                kd[[0, 1, 2, 6, 8, 9, 10, 11, 12, 13]] *= self.kd_ratios
                kd[[3, 4, 5, 7, 14, 15, 16, 17, 18, 19]] *= self.kd_ratios

            if len(self.kd_ratios) == 7:
                kd[0:8] *= self.kd_ratios[0]
                kd[8] *= self.kd_ratios[1]
                kd[9] *= self.kd_ratios[2]
                kd[10] *= self.kd_ratios[3]
                kd[11] *= self.kd_ratios[4]
                kd[12] *= self.kd_ratios[5]
                kd[13] *= self.kd_ratios[6]
                kd[14] *= self.kd_ratios[1]
                kd[15] *= self.kd_ratios[2]
                kd[16] *= self.kd_ratios[3]
                kd[17] *= self.kd_ratios[4]
                kd[18] *= self.kd_ratios[5]
                kd[19] *= self.kd_ratios[6]

            tau = -kp * (q[self.actuated_dof_ids] - self.target) - kd * qdot[self.actuated_dof_ids]

            if self.limited_joint_vel:
                tau[(np.abs(np.array(self.robot_skeleton.dq)[self.actuated_dof_ids]) > self.joint_vel_limit) * (
                        np.sign(np.array(self.robot_skeleton.dq))[self.actuated_dof_ids] == np.sign(tau))] = 0
        else:
            raise NotImplementedError

        torqs = self.ClampTorques(tau)

        return torqs

    def ClampTorques(self, torques):
        torqueLimits = self.torqueLimits

        for i in range(len(torques)):
            if torques[i] > torqueLimits:  #
                torques[i] = torqueLimits
            if torques[i] < -torqueLimits:
                torques[i] = -torqueLimits

        return torques

    def get_ref_pose(self, t):
        ref_target = self.interp_sch[0][1]

        for i in range(len(self.interp_sch) - 1):
            if t >= self.interp_sch[i][0] and t < self.interp_sch[i + 1][0]:
                ratio = (t - self.interp_sch[i][0]) / (self.interp_sch[i + 1][0] - self.interp_sch[i][0])
                ref_target = ratio * self.interp_sch[i + 1][1] + (1 - ratio) * self.interp_sch[i][1]
        if t > self.interp_sch[-1][0]:
            ref_target = self.interp_sch[-1][1]
        return ref_target

    def step(self, a):
        print("begin step")
        self.current_pd = self.assist_schedule[0][1][0]
        if len(self.assist_schedule) > 0:
            for sch in self.assist_schedule:
                if self.t > sch[0]:
                    self.current_pd = sch[1][0]

        if self.use_discrete_action:
            # a = a * 1.0 / np.floor(self.action_space.nvec / 2.0) - 1.0
            new_a = np.zeros(len(a))
            odd_nvec_ids = self.action_space.nvec % 2 == 1
            even_nvec_ids = self.action_space.nvec % 2 == 0
            new_a[odd_nvec_ids] = a[odd_nvec_ids] * 1.0/ np.floor(self.action_space.nvec[odd_nvec_ids]/2.0) - 1.0
            new_a[even_nvec_ids] = a[even_nvec_ids] * 1.0 / (np.floor(self.action_space.nvec[even_nvec_ids] / 2.0)-0.5) - 1.0
            a = new_a

        action_bound_violation = np.sum(np.clip(np.abs(a) - 1, 0, 10000))

        # if not self.butterworth_filter:
        #     self.action_filter_cache.append(a)
        #     if len(self.action_filter_cache) > self.action_filtering:
        #         self.action_filter_cache.pop(0)
        #     if self.action_filtering > 0:
        #         a = np.mean(self.action_filter_cache, axis=0)
            # self.action_buffer.append(np.copy(a))

        # modify gravity according to schedule
        grav = self.gravity_sch[0][1]

        for i in range(len(self.gravity_sch) - 1):
            if self.t >= self.gravity_sch[i][0] and self.t < self.gravity_sch[i + 1][0]:
                ratio = (self.t - self.gravity_sch[i][0]) / (self.gravity_sch[i + 1][0] - self.gravity_sch[i][0])
                grav = ratio * self.gravity_sch[i + 1][1] + (1 - ratio) * self.gravity_sch[i][1]
        if self.t > self.gravity_sch[-1][0]:
            grav = self.gravity_sch[-1][1]
        self.dart_world.set_gravity(grav)

        xpos_before = self.robot_skeleton.q[3]
        self.advance(a)
        xpos_after = self.robot_skeleton.q[3]

        upright_rew = np.abs(euler_from_matrix(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3], 'sxyz')[1])


        # if xpos_after - self.init_x_progress < 0.15:
        #     self.alive_bonus -= self.init_alive_bonus / 40
        # else:
        self.alive_bonus = self.init_alive_bonus
        self.init_x_progress = xpos_after

        pose_math_rew = np.sum(
            np.abs(np.array(self.ref_target - np.array(self.robot_skeleton.q)[self.actuated_dof_ids])) ** 2)
        reward = -self.energy_weight * np.sum(
            self.tau ** 2) + self.alive_bonus - pose_math_rew * self.pose_weight
        reward -= self.work_weight * np.sum(np.abs(self.tau * np.array(self.robot_skeleton.dq)[self.actuated_dof_ids]))
        reward -= 2.0 * np.abs(self.robot_skeleton.dC[1])
        reward -= self.upright_weight * upright_rew
        reward -= 0.3 * action_bound_violation

        # if falling to the back, encourage the robot to take steps backward
        current_forward_reward = self.forward_reward
        reward += current_forward_reward * np.clip((xpos_after - xpos_before) / self.dt, -self.velocity_clip,
                                                self.velocity_clip)

        reward -= self.comvel_pen * np.linalg.norm(self.robot_skeleton.dC)
        reward -= self.compos_pen * np.linalg.norm(self.init_q[3:6] - self.robot_skeleton.q[3:6])

        self.reward_terms[0] += -self.energy_weight *np.sum(self.tau ** 2)
        self.reward_terms[1] += -self.work_weight*np.sum(np.abs(self.tau * np.array(self.robot_skeleton.dq)[self.actuated_dof_ids]))
        self.reward_terms[2] += -2*np.abs(self.robot_skeleton.dC[1])
        self.reward_terms[3] += -self.upright_weight * upright_rew
        self.reward_terms[4] += -0.3*action_bound_violation
        self.reward_terms[5] += current_forward_reward * np.clip((xpos_after - xpos_before) / self.dt, -self.velocity_clip, self.velocity_clip)
        self.reward_terms[6] += pose_math_rew * self.pose_weight

        low_lim_violate = np.sum(np.array(self.robot_skeleton.q)[14:] - np.array(self.robot_skeleton.q_lower)[14:] <= 0.01)
        high_lim_violate = np.sum(np.array(self.robot_skeleton.q)[14:] - np.array(self.robot_skeleton.q_upper)[14:] >= -0.01)

        reward -= (low_lim_violate + high_lim_violate)
        # print(low_lim_violate + high_lim_violate)

        self.reward_terms[7] += -(low_lim_violate + high_lim_violate)
        self.reward_terms[8] += self.alive_bonus

        s = self.state_vector()
        done = not (np.isfinite(s['qvel']).all() and (np.abs(s['qvel']) < 200).all())

        if np.any(np.abs(np.array(self.robot_skeleton.q)[0:2]) > self.orientation_threshold):
            if DEBUG:
                print("orientation threshold violated")
            done = True

        if not self.side_walk and np.abs(np.array(self.robot_skeleton.q)[2]) > self.orientation_threshold:
            if DEBUG:
                print("orientation threshold violated")
            done = True

        self.fall_on_ground = False
        self_colliding = False
        contacts = self.dart_world.collision_result.contacts
        total_force = np.zeros(3)

        ground_bodies = [self.dart_world.skeletons[0].bodynodes[0]]

        feet_in_contact = [False, False]
        for contact in contacts:
            if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                continue
            if contact.bodynode1 in ground_bodies or contact.bodynode2 in ground_bodies:
                total_force += contact.force
            if contact.bodynode1 not in self.permitted_contact_bodies and contact.bodynode2 not in self.permitted_contact_bodies:
                if contact.bodynode1 in ground_bodies or contact.bodynode2 in ground_bodies:
                    self.fall_on_ground = True
            if contact.bodynode1.skel == contact.bodynode2.skel and self.cur_step > 1:
                self_colliding = True
            if contact.bodynode1.id == self.left_foot_id or contact.bodynode2.id == self.left_foot_id and not self_colliding:
                feet_in_contact[0] = True
            if contact.bodynode1.id == self.right_foot_id or contact.bodynode2.id == self.right_foot_id and not self_colliding:
                feet_in_contact[1] = True
        if self.terminate_sliding_contact:
            # import pdb; pdb.set_trace()
            if feet_in_contact[0]:
                if self.foot_contact_buffer[0] is None:
                    self.foot_contact_buffer[0] = [self.t, self.robot_skeleton.bodynodes[self.left_foot_id].C[0]]
                else:
                    if self.t - self.foot_contact_buffer[0][0] > 0.3 and np.abs(self.robot_skeleton.bodynodes[self.left_foot_id].C[0] - self.foot_contact_buffer[0][1]) > 0.03:
                        if DEBUG:
                            print("sliding problem")
                        done = True
                    if self.task_mode == self.STEPPING and self.t - self.foot_contact_buffer[0][0] > 1.0:
                        if DEBUG:
                            print("sliding problem")
                        done = True
            else:
                self.foot_contact_buffer[0] = None

            if feet_in_contact[1]:
                if self.foot_contact_buffer[1] is None:
                    self.foot_contact_buffer[1] = [self.t, self.robot_skeleton.bodynodes[self.right_foot_id].C[0]]
                else:
                    if self.t - self.foot_contact_buffer[1][0] > 0.3 and np.abs(self.robot_skeleton.bodynodes[self.right_foot_id].C[0] - self.foot_contact_buffer[1][1]) > 0.03:
                        if DEBUG:
                            print("sliding problem")
                        done = True
                    if self.task_mode == self.STEPPING and self.t - self.foot_contact_buffer[1][0] > 1.0:
                        if DEBUG:
                            print("sliding problem")
                        done = True
            else:
                self.foot_contact_buffer[1] = None

        if self.t > self.interp_sch[-1][0] + 2:
            if DEBUG:
                print("ref traj time exceeded")
            done = True

        if self.fall_on_ground:
            if DEBUG:
                print("fall on ground")
            done = True

        if self_colliding:
            if DEBUG:
                print("self colliding")
            done = True

        if self.init_q[5] - self.robot_skeleton.q[5] > self.height_drop_threshold:
            if DEBUG:
                print("height too low")
            done = True

        # action acceleration
        action_accel = self.action_filter_cache[-1] + self.action_filter_cache[-3] - self.action_filter_cache[-2]*2
        if np.max(np.abs(action_accel)) > 1.0:
            if DEBUG:
                print("action acceleration threshold violated")
            done = True

        if self.compos_range > 0:
            if self.forward_reward == 0:
                if np.linalg.norm(self.init_q[4:6] - self.robot_skeleton.q[4:6]) > self.compos_range:
                    if DEBUG:
                        print("compos range")
                    done = True
            else:
                if np.linalg.norm(self.init_q[4:6] - self.robot_skeleton.q[4:6]) > self.compos_range:
                    if DEBUG:
                        print("com position range exceeded")
                    done = True

        if self.task_mode == self.BONGOBOARD:
            reward -= 10.0 * np.abs(self.dart_world.skeletons[4].q[0])
            board_touching_ground = False
            for contact in contacts:
                if contact.bodynode1 in ground_bodies or contact.bodynode2 in ground_bodies:
                    if contact.bodynode1 == self.dart_world.skeletons[4].bodynodes[1] or contact.bodynode2 == self.dart_world.skeletons[4].bodynodes[1]:
                        board_touching_ground = True
            if board_touching_ground:
                reward -= 5

        reward -= self.contact_pen * np.linalg.norm(total_force) # penalize contact forces

        if self.task_mode == self.WALK and 0.25 < np.linalg.norm(self.robot_skeleton.bodynode('MP_ANKLE2_R').C - self.robot_skeleton.bodynode('MP_ANKLE2_L').C):
            done = True

        if self.t > 2.0 and np.abs(self.robot_skeleton.C[0]) < 0.15 and self.forward_reward != 0.0:
            if DEBUG:
                print("no move termination")
            done = True

        if done:
            reward = 0


        # if np.abs(self.robot_skeleton.C[0]) > 0.1:
        #     done = True

        ob = self._get_obs(update_buffer=True)

        # move the obstacle forward when the robot has passed it
        horizontal_range = [-0.0, 0.0]
        if self.randomize_obstacle and not self.soft_ground and not self.supress_all_randomness:
            vertical_range = [-1.363, -1.373]
            if self.variation_scheduling is not None and 'obstacle_height_range' in self.variation_scheduling[0][1]:
                height_var = self.variation_scheduling[0][1]['obstacle_height_range']
                for sch_id in range(len(self.variation_scheduling)):
                    if self.robot_skeleton.q[3] > self.variation_scheduling[sch_id][0]:
                        height_var = self.variation_scheduling[sch_id][1]['obstacle_height_range']
                vertical_range = [-1.368 - height_var, -1.368 + height_var]
        else:
            vertical_range = [-1.398, -1.398]
        for obid in range(1, len(self.obstacle_bodynodes)):
            if self.robot_skeleton.C[0] - 0.35 > self.obstacle_bodynodes[obid].shapenodes[0].offset()[0]:
                last_ob_id = (obid-1 + len(self.obstacle_bodynodes)-2) % (len(self.obstacle_bodynodes)-1)+1
                last_ob_pos = self.obstacle_bodynodes[last_ob_id].shapenodes[0].offset()[0]
                offset = np.copy(self.obstacle_bodynodes[obid].shapenodes[0].offset())

                sampled_v = np.random.uniform(vertical_range[0], vertical_range[1])
                sampled_h = np.random.uniform(horizontal_range[0], horizontal_range[1])

                offset[0] = last_ob_pos + 0.1 + sampled_h
                offset[2] = sampled_v

                self.obstacle_bodynodes[obid].shapenodes[0].set_offset(offset)
                self.obstacle_bodynodes[obid].shapenodes[1].set_offset(offset)

        #if self.range_robust > 0:
        #    rand_param = np.clip(self.current_param + np.random.normal(0, self.range_robust, len(self.current_param)), -0.05, 1.05)
        #    self.param_manager.set_simulator_parameters(rand_param)

        info = {}
        if self.variation_scheduling is not None:
            curriculum_progress = self.robot_skeleton.q[3] / self.variation_scheduling[1][0]
            info['curriculum_progress'] = curriculum_progress
        # if done or self.cur_step > 499:
        #     print(self.reward_terms)
        print("end step")
        return ob, reward, done, info

    def _get_obs(self, update_buffer=False):
        if update_buffer:
            state = np.concatenate([np.array(self.robot_skeleton.q)[self.observed_dof_ids], np.array(self.robot_skeleton.dq)[self.observed_dof_ids]])
            if self.multipos_obs > 0:
                if self.streaming_mode:
                    state = self.state_cache[0]
                else:
                    state = np.array(self.robot_skeleton.q)[self.observed_dof_ids]

                self.obs_cache.append(state)
                while len(self.obs_cache) < self.multipos_obs:
                    self.obs_cache.append(state)
                if len(self.obs_cache) > self.multipos_obs:
                    self.obs_cache.pop(0)

                for i in range(len(self.obs_cache)-1):
                    if self.input_obs_difference:
                        if self.input_difference_sign:
                            state = np.concatenate([np.sign(self.obs_cache[i] - self.obs_cache[-1]), state])
                        else:
                            state = np.concatenate([(self.obs_cache[i] - self.obs_cache[-1])/self.dt, state])
                    else:
                        state = np.concatenate([self.obs_cache[i], state])

            for i in range(self.future_ref_pose):
                state = np.concatenate([state, self.get_ref_pose(self.t + self.dt * (i+1))])

            if self.gyro_only_mode:
                state = np.array([])

            if self.root_input:
                if self.streaming_mode:
                    gyro = self.gyro_cache[0]
                else:
                    gyro = self.get_sim_bno55()
                if not self.include_heading:
                    gyro = np.array([gyro[0], gyro[1], self.last_root[0], self.last_root[1]])
                    self.last_root = [gyro[0], gyro[1]]
                else:
                    adjusted_heading = (gyro[2] - self.initial_heading) % (2*np.pi)
                    adjusted_heading = adjusted_heading - 2*np.pi if adjusted_heading > np.pi else adjusted_heading
                    gyro = np.array([gyro[0], gyro[1], adjusted_heading, self.last_root[0], self.last_root[1], self.last_root[2]])
                    if self.use_zeroed_gyro:
                        gyro[0:2] -= self.initial_gyro[0:2]
                        gyro[0:2] += self.gyro_bias
                    self.last_root = [gyro[0], gyro[1], adjusted_heading]
                state = np.concatenate([state, gyro])

            if self.include_base_linear_info:
                state = np.concatenate([state, self.robot_skeleton.bodynode('MP_BODY').com_spatial_velocity()[3:6]])

            if self.noisy_input and not self.supress_all_randomness:
                state = state + np.random.normal(0, .01, len(state))
            if self.fallstate_input:
                state = np.concatenate([state, self.falling_state()])

            if self.transition_input:
                if self.t < 1.0:
                    state = np.concatenate([state, [0]])
                else:
                    state = np.concatenate([state, [1]])

            if self.task_mode == self.BONGOBOARD:
                board_ori = np.array(euler_from_matrix(self.dart_world.skeletons[4].bodynode('board').T[0:3, 0:3], 'sxyz'))
                state = np.concatenate([state, board_ori])

            if self.input_leg_tracking_error:
                cur_leg_pose = np.array(self.robot_skeleton.q)[self.observed_dof_ids]
                target_leg_pose = self.target[np.array(self.observed_dof_ids)-6]
                leg_tracking_index = len(state)
                state = np.concatenate([state, (cur_leg_pose)])

            if self.input_contact_bin:
                contacts = self.dart_world.collision_result.contacts
                left_foot_contact_bin = np.zeros(self.num_contact_bin)
                right_foot_contact_bin = np.zeros(self.num_contact_bin)

                for contact in contacts:
                    if contact.skel_id1 != self.robot_skeleton.id and contact.skel_id2 != self.robot_skeleton.id:
                        continue
                    for shoe_id in range(len(self.left_foot_shoe_bodies)):
                        if contact.bodynode1 == self.left_foot_shoe_bodies[shoe_id] or contact.bodynode2 == self.left_foot_shoe_bodies[shoe_id]:
                            left_foot_contact_bin[shoe_id] = 1.0
                    for shoe_id in range(len(self.right_foot_shoe_bodies)):
                        if contact.bodynode1 == self.right_foot_shoe_bodies[shoe_id] or contact.bodynode2 == \
                                self.right_foot_shoe_bodies[shoe_id]:
                            right_foot_contact_bin[shoe_id] = 1.0

                for bin_id in range(self.num_contact_bin):
                    if left_foot_contact_bin[bin_id] == 1:
                        self.left_foot_shoe_bodies[bin_id].shapenodes[0].set_visual_aspect_rgba([0.5, 0.5, 1.0, 1.0])
                    else:
                        self.left_foot_shoe_bodies[bin_id].shapenodes[0].set_visual_aspect_rgba([0.9, 0.9, 0.0, 1.0])
                    if right_foot_contact_bin[bin_id] == 1:
                        self.right_foot_shoe_bodies[bin_id].shapenodes[0].set_visual_aspect_rgba([0.5, 0.5, 1.0, 1.0])
                    else:
                        self.right_foot_shoe_bodies[bin_id].shapenodes[0].set_visual_aspect_rgba([0.9, 0.9, 0.0, 1.0])

                state = np.concatenate([state, left_foot_contact_bin, right_foot_contact_bin])

            if self.input_range_sensor:
                left_foot_range = np.zeros(self.range_sensor_param[1])
                right_foot_range = np.zeros(self.range_sensor_param[1])

                foot_samples_local = (np.arange(self.range_sensor_param[1]) / self.range_sensor_param[1] - 0.5) * self.range_sensor_param[0] * 2
                left_foot_samples = [self.body_parts[-7].to_world([0, 0, p]) for p in foot_samples_local]
                right_foot_samples = [self.body_parts[-1].to_world([0, 0, p]) for p in foot_samples_local]

                obstacle_width = self.obstacle_bodynodes[1].shapenodes[0].shape.size()[0]
                for obid in range(len(self.obstacle_bodynodes)):
                    obs_pos = self.obstacle_bodynodes[obid].shapenodes[0].offset()
                    obs_pos[2] += self.obstacle_bodynodes[obid].shapenodes[0].shape.size()[2] / 2.0
                    for l in range(len(left_foot_samples)):
                        if left_foot_samples[l][0] > obs_pos[0] - obstacle_width / 2.0 and\
                            left_foot_samples[l][0] < obs_pos[0] + obstacle_width / 2.0:
                            left_foot_range[l] = left_foot_samples[l][2] - obs_pos[2]
                    for l in range(len(right_foot_samples)):
                        if right_foot_samples[l][0] > obs_pos[0] - obstacle_width / 2.0 and\
                            right_foot_samples[l][0] < obs_pos[0] + obstacle_width / 2.0:
                            right_foot_range[l] = right_foot_samples[l][2] - obs_pos[2]
                left_foot_range = np.clip(left_foot_range, 0.0, 0.2)
                right_foot_range = np.clip(right_foot_range, 0.0, 0.2)
                state = np.concatenate([state, left_foot_range, right_foot_range])

            if self.adjustable_leg_compliance:
                state = np.concatenate([state, [self.left_leg_compliance, self.right_leg_compliance]])

            if self.train_UP:
                #UP = self.param_manager.get_simulator_parameters()
                state = np.concatenate([state, self.current_param])

            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                if i > 0 and self.input_obs_difference:
                    if self.input_difference_sign:
                        final_obs = np.concatenate([final_obs, np.sign(self.observation_buffer[-self.obs_delay - 1 - i] -
                                                    self.observation_buffer[-self.obs_delay - 1])])
                    else:
                        final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay - 1 - i] - self.observation_buffer[-self.obs_delay - 1]])
                else:
                    hist_obs = self.observation_buffer[-self.obs_delay - 1 - i]
                    if self.input_leg_tracking_error:
                        target_leg_pose = self.target[np.array(self.observed_dof_ids) - 6]
                        # Use the current target to offset the observed leg angles, to match real world case
                        hist_obs[leg_tracking_index:leg_tracking_index+len(target_leg_pose)] -= target_leg_pose
                    final_obs = np.concatenate([final_obs, hist_obs])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0] * 0.0])

        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[- 1 - i]])
            else:
                final_obs = np.concatenate([final_obs, np.zeros(self.act_dim)])
        return final_obs

    def reset_model(self):
        self.dart_world.reset()
        qpos = np.zeros(self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq
        if not self.supress_all_randomness:
            qpos += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
            qvel += self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        # LEFT HAND
        if self.interp_sch is not None:
            qpos[self.actuated_dof_ids] = np.clip(self.interp_sch[0][1], JOINT_LOW_BOUND, JOINT_UP_BOUND)
        else:
            qpos[self.actuated_dof_ids] = np.clip(0.5 * (pose_squat_rad + pose_stand_rad), JOINT_LOW_BOUND, JOINT_UP_BOUND)

        self.count = 0
        qpos[0:6] += self.init_root_pert

        if self.initialize_falling and not self.supress_all_randomness:
            qvel[0] = np.random.uniform(-2.0, 2.0)
            qvel[1] = np.random.uniform(-2.0, 2.0)

        self.set_state(qpos, qvel)

        q = self.robot_skeleton.q

        qid = 5
        if self.task_mode == self.CRAWL:
            q[qid] += -0.3 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2]])
        elif self.task_mode == self.BONGOBOARD:
            q[qid] += -0.25 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2]])
        else:
            if not self.soft_foot:
                q[qid] += -0.335 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2]])
            else:
                q[qid] += -0.335 - np.min([self.body_parts[-1].C[2], self.body_parts[-7].C[2], self.left_foot_shoe_bodies[0].C[2], self.right_foot_shoe_bodies[0].C[2]])

        if self.use_settled_initial_states:
            q = self.init_states_candidates[np.random.randint(len(self.init_states_candidates))]

        self.robot_skeleton.q = q

        if self.initial_state_set is not None:
            init_s_id = np.random.randint(len(self.initial_state_set))
            self.set_state_vector(self.initial_state_set[init_s_id])

        self.init_q = np.copy(self.robot_skeleton.q)

        self.initial_gyro = self.get_sim_bno55(supress_randomization=True)

        self.t = 0
        self.last_root = [0, 0]
        if self.include_heading:
            self.last_root = [0, 0, 0]
            self.initial_heading = self.get_sim_bno55()[2]

        self.observation_buffer = []
        self.action_buffer = []
        self.target_pose_history = []

        if self.resample_MP and not self.supress_all_randomness:
            if self.precomputed_dynamics_parameters_set is None:
                self.param_manager.resample_parameters()
                self.current_param = self.param_manager.get_simulator_parameters()
            else:
                self.current_param = self.precomputed_dynamics_parameters_set[
                    np.random.randint(0, len(self.precomputed_dynamics_parameters_set))]
                self.param_manager.set_simulator_parameters(self.current_param)
            if self.range_robust > 0:
                lb = np.clip(self.current_param - self.range_robust, -0.05, 1.05)
                ub = np.clip(self.current_param + self.range_robust, -0.05, 1.05)
                self.current_param = np.random.uniform(lb, ub)

        if self.randomize_timestep and not self.supress_all_randomness:
            new_control_dt = self.control_interval + np.random.uniform(0.0, 0.01)
            default_fs = int(self.control_interval / self.sim_timestep)
            if not self.gyro_only_mode:
                self.frame_skip = np.random.randint(-5, 5) + default_fs

            self.dart_world.dt = new_control_dt / self.frame_skip

        self.obs_cache = []
        if self.resample_MP or self.mass_ratios[0] != 0:
            for i in range(len(self.body_parts)):
                self.body_parts[i].set_mass(self.orig_bodynode_masses[i] * self.mass_ratios[i])
            for i in range(len(self.body_parts)):
                self.body_parts[i].set_inertia(self.orig_bodynode_inertias[i] * self.inertia_ratios[i])

        self.dart_world.skeletons[2].q = [0,0,0, 100, 100, 100]

        if self.randomize_gravity_sch and not self.supress_all_randomness:
            self.gravity_sch = [[0.0, np.array([0,0,-9.81])]] # always start from normal gravity
            num_change = np.random.randint(1, 3) # number of gravity changes
            interv = self.interp_sch[-1][0] / num_change
            for i in range(num_change):
                rots = np.random.uniform(-0.5, 0.5, 2)
                self.gravity_sch.append([(i+1) * interv, np.array([np.cos(rots[0])*np.sin(rots[1]), np.sin(rots[0]), -np.cos(rots[0])*np.cos(rots[1])]) * 9.81])

        horizontal_range = [-0.0, 0.0]
        if self.randomize_obstacle and not self.soft_ground and not self.supress_all_randomness:
            vertical_range = [-1.363, -1.373]
            if self.variation_scheduling is not None and 'obstacle_height_range' in self.variation_scheduling[0][1]:
                vertical_range = [-1.368 - self.variation_scheduling[0][1]['obstacle_height_range'], -1.368 + self.variation_scheduling[0][1]['obstacle_height_range']]
        else:
            vertical_range = [-1.398, -1.398]
            self.obstacle_bodynodes[0].shapenodes[0].set_offset([0, 0, -1.4])
        for obid in range(1, len(self.obstacle_bodynodes)):
            sampled_v = np.random.uniform(vertical_range[0], vertical_range[1])
            sampled_h = np.random.uniform(horizontal_range[0], horizontal_range[1]) + 0.05 + 0.1 * obid
            self.obstacle_bodynodes[obid].shapenodes[0].set_offset([sampled_h, 0, sampled_v])
            self.obstacle_bodynodes[obid].shapenodes[1].set_offset([sampled_h, 0, sampled_v])

        if self.randomize_gyro_bias and not self.supress_all_randomness:
            self.gyro_bias = np.random.uniform(-0.1, 0.1, 2)

        if self.randomize_action_delay and not self.supress_all_randomness:
            self.action_delay = np.random.uniform(0.01, 0.03)

        self.perturb_force = np.array([0.0, 0.0, 0.0])
        self.cur_step = 0

        self.state_cache = [np.array(self.robot_skeleton.q)[6:], 0.0, 0.015 + np.random.normal(0.0, 0.005)]
        self.gyro_cache = [self.get_sim_bno55(), 0.0, 0.015 + np.random.normal(0.0, 0.005)]

        self.action_queue = []
        self.target = self.init_q[self.actuated_dof_ids]

        if self.butterworth_filter:
            self.action_filter.reset_filter(self.target)

        self.action_filter_cache = []
        for i in range(self.action_filtering):
            self.action_filter_cache.append(self.target)

        self.left_leg_compliance = 1
        self.right_leg_compliance = 1

        self.foot_contact_buffer = [None, None]

        self.reward_terms = [0.0] * 9

        self.alive_bonus = self.init_alive_bonus
        self.init_x_progress = self.robot_skeleton.C[0]

        return self._get_obs(update_buffer=True)

    def resample_task(self):
        self.resample_MP = False

        if self.precomputed_dynamics_parameters_set is None:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()
        else:
            self.current_param = self.precomputed_dynamics_parameters_set[np.random.randint(0, len(self.precomputed_dynamics_parameters_set))]
            self.param_manager.set_simulator_parameters(self.current_param)

        return np.copy(self.current_param)

    def set_task(self, task_params):
        self.param_manager.set_simulator_parameters(task_params)


    def get_robot_optimization_setup(self):
        # optimize the shape
        # dim = 10
        # upper_bound = np.ones(dim) * 0.01
        # lower_bound = -np.ones(dim) * 0.01

        # optimize the spring stiffness
        dim = 10
        upper_bound = np.ones(dim) * 4000
        lower_bound = np.ones(dim) * 300
        return dim, upper_bound, lower_bound

    def set_robot_optimization_parameters(self, parameters):
        # set offset of the shoe
        assert(len(parameters) == 10)
        # for i in range(10):
        #     for sn in self.robot_skeleton.bodynode('SHOE_PIECE'+str(i+1)+'_L').shapenodes:
        #         sn.set_offset([parameters[i], 0.0, 0.0])
        #     for sn in self.robot_skeleton.bodynode('SHOE_PIECE'+str(i+1)+'_R').shapenodes:
        #         sn.set_offset([-parameters[i], 0.0, 0.0])
        for i in range(len(self.left_shoe_dofs)):
            self.left_shoe_dofs[i].set_spring_stiffness(parameters[i])
        for i in range(len(self.right_shoe_dofs)):
            self.right_shoe_dofs[i].set_spring_stiffness(parameters[i])


    def viewer_setup(self):
        if not self.disableViewer:
            #self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -1.0
            self._get_viewer().scene.tb.trans[1] = 0.0
            self._get_viewer().scene.tb.theta = 80
            self._get_viewer().scene.tb.phi = 0

        return 0

    def advance_curriculum(self):
        self.variation_scheduling[0][1]['obstacle_height_range'] = self.variation_scheduling[1][1]['obstacle_height_range']
        self.variation_scheduling[1][1]['obstacle_height_range'] += 0.001
        if self.variation_scheduling[1][1]['obstacle_height_range'] > 0.01:
            self.variation_scheduling[1][1]['obstacle_height_range'] -= 0.001

    def get_curriculum(self):
        return self.variation_scheduling[0][1]['obstacle_height_range']

    def state_vector(self):
        if len(self.observation_buffer) > 10:
            obs_buffer = np.array(self.observation_buffer[-10:])
        else:
            obs_buffer = np.array(self.observation_buffer)
        if len(self.action_buffer) > 10:
            act_buffer = np.array(self.action_buffer[-10:])
        else:
            act_buffer = np.array(self.action_buffer)
        return {'qpos':np.array(self.robot_skeleton.q), 'qvel':np.array(self.robot_skeleton.dq), 't':self.t,
                'pose_cache':np.array(self.obs_cache), 'obs_buffer':obs_buffer, 'act_buffer': act_buffer,
                'act_queue':self.action_queue, 'action_filter_cache':np.array(self.action_filter_cache)}

    def set_state_vector(self, s):
        self.robot_skeleton.q = s['qpos']
        self.robot_skeleton.dq = s['qvel']
        self.t = s['t']
        self.obs_cache = list(copy.deepcopy(s['pose_cache']))
        self.observation_buffer = list(copy.deepcopy(s['obs_buffer']))
        self.action_buffer = list(copy.deepcopy(s['act_buffer']))
        self.action_queue = list(copy.deepcopy(s['act_queue']))
        self.action_filter_cache = list(copy.deepcopy(s['action_filter_cache']))

        # self.reset_filtering()

    def reset_filtering(self):
        target = self.get_ref_pose(self.t)

        if self.butterworth_filter:
            self.action_filter.reset_filter(target)

        self.action_filter_cache = []
        for i in range(self.action_filtering):
            self.action_filter_cache.append(target)

