__author__ = 'yuwenhao'

import numpy as np
from gym.envs.dart import dart_env

##############################################################################################################
################################  Hopper #####################################################################
##############################################################################################################



class hopperContactMassManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.1, 1.0] # friction range
        self.restitution_range = [0.0, 0.3]
        self.mass_range = [2.0, 15.0]
        self.damping_range = [0.5, 3.0]
        self.power_range = [100, 300]
        self.ankle_range = [60, 300]
        self.velrew_weight_range = [-1.0, 1.0]
        self.com_offset_range = [-0.05, 0.05]
        self.frame_skip_range = [4, 10]
        self.actuator_nonlin_range = [0.75, 1.5]
        self.reward_predictor = None

        self.activated_param = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#[0, 1, 2, 5, 9]#[0,1, 2,3,4,5, 6,7,8, 9, 12,13,14,15]
        self.controllable_param = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#[0, 1, 2, 5, 9]#[0,1, 2,3,4,5, 6,7,8, 9, 12,13,14,15]

        self.binned_param = 0 # don't bin if = 0

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        mass_param = []
        for bid in range(2, 6):
            cur_mass = self.simulator.robot_skeleton.bodynodes[bid].m
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 6):
            cur_damp = self.simulator.robot_skeleton.joints[jid].damping_coefficient(0)
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))

        cur_power = self.simulator.action_scale
        power_param = (cur_power[0] - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        if self.simulator.pid_controller is not None:
            ank_power_param = 1.0
        else:
            cur_ank_power = self.simulator.action_scale[2]
            ank_power_param = (cur_ank_power - self.ankle_range[0]) / (self.ankle_range[1] - self.ankle_range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (self.velrew_weight_range[1] - self.velrew_weight_range[0])

        com_param = []
        for bid in range(2, 6):
            if bid != 5:
                cur_com = self.simulator.robot_skeleton.bodynodes[bid].local_com()[1] - self.simulator.initial_local_coms[bid][1]
            else:
                cur_com = self.simulator.robot_skeleton.bodynodes[bid].local_com()[0] - self.simulator.initial_local_coms[bid][0]
            com_param.append((cur_com - self.com_offset_range[0]) / (self.com_offset_range[1] - self.com_offset_range[0]))

        cur_frameskip = self.simulator.frame_skip
        frameskip_param = (cur_frameskip - self.frame_skip_range[0]) / (self.frame_skip_range[1] - self.frame_skip_range[0])

        cur_act_nonlin = self.simulator.actuator_nonlin_coef
        act_nonlin_param = (cur_act_nonlin - self.actuator_nonlin_range[0]) / (self.actuator_nonlin_range[1] - self.actuator_nonlin_range[0])

        params = np.array([friction_param, restitution_param]+ mass_param + damp_param +
                          [power_param, ank_power_param, velrew_param] + com_param + [frameskip_param, act_nonlin_param])[self.activated_param]
        if self.binned_param > 0:
            for i in range(len(params)):
                params[i] = int(params[i] / (1.0 / self.binned_param)) * (1.0/self.binned_param) + 0.5 / self.binned_param
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 1 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        for bid in range(2, 6):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.robot_skeleton.bodynodes[bid].set_mass(mass)
                cur_id += 1
        for jid in range(6, 9):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.robot_skeleton.joints[jid - 3].set_damping_coefficient(0, damp)
                cur_id += 1
        if 9 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.action_scale[0:3] = np.array([power, power, power])
            cur_id += 1
        if 10 in self.controllable_param:
            ankpower = x[cur_id] * (self.ankle_range[1] - self.ankle_range[0]) + self.ankle_range[0]
            self.simulator.action_scale[2] = ankpower
            cur_id += 1
        if 11 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1

        for bid in range(2, 6):
            if bid+10 in self.controllable_param:
                com = x[cur_id] * (self.com_offset_range[1] - self.com_offset_range[0]) + self.com_offset_range[0]
                init_com = np.copy(self.simulator.initial_local_coms[bid])
                if bid != 5:
                    init_com[1] += com
                else:
                    init_com[0] += com
                self.simulator.robot_skeleton.bodynodes[bid].set_local_com(init_com)
                cur_id += 1

        if 16 in self.controllable_param:
            frame_skip = x[cur_id] * (self.frame_skip_range[1] - self.frame_skip_range[0]) + self.frame_skip_range[0]
            self.simulator.frame_skip = int(frame_skip)
            cur_id += 1

        if 17 in self.controllable_param:
            act_nonlin = x[cur_id] * (self.actuator_nonlin_range[1] - self.actuator_nonlin_range[0]) + self.actuator_nonlin_range[0]
            self.simulator.actuator_nonlin_coef = act_nonlin
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.reward_predictor is not None:
            while True:
                pred_rew = self.reward_predictor.predict_reward(x)[0]
                normed_rew = (pred_rew - self.reward_predictor.reward_range[0]) / (self.reward_predictor.reward_range[1] - self.reward_predictor.reward_range[0])
                normed_rew = np.clip(normed_rew, 0.05, 0.95)
                if np.random.random() > normed_rew:
                    break
                else:
                    x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
            print(normed_rew, x)
        self.set_simulator_parameters(x)


class mjHopperManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.2, 1.0]  # friction range
        self.mass_range = [2.0, 7.0]
        self.damping_range = [0.15, 2.0]
        self.power_range = [150, 500]
        self.velrew_weight_range = [-1.0, 1.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.ankle_jnt_range = [0.5, 1.0]

        self.activated_param = [1]#[0, 1,2,3,4, 5,6,7, 8, 10, 11, 12, 13]
        self.controllable_param = [1]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        mass_param = []
        for bid in range(1, 5):
            cur_mass = self.simulator.model.body_mass[bid]
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 6):
            cur_damp = self.simulator.model.dof_damping[jid]
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))

        cur_power = self.simulator.model.actuator_gear[0][0]
        power_param = (cur_power - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (
                self.velrew_weight_range[1] - self.velrew_weight_range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        cur_jntlimit = self.simulator.model.jnt_range[-1][0]
        jntlimit_param = (cur_jntlimit - self.ankle_jnt_range[0]) / (self.ankle_jnt_range[1] - self.ankle_jnt_range[0])

        params = np.array([friction_param] + mass_param + damp_param + [power_param, velrew_param, rest_param
                                                                        ,solimp_param, solref_param, armature_param,
                                                                        jntlimit_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.model.geom_friction[-1][0] = friction
            cur_id += 1
        for bid in range(1, 5):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.model.body_mass[bid] = mass
                cur_id += 1
        for jid in range(5, 8):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.model.dof_damping[jid - 2] = damp
                cur_id += 1
        if 8 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.model.actuator_gear[0][0] = power
            self.simulator.model.actuator_gear[1][0] = power
            self.simulator.model.actuator_gear[2][0] = power
            cur_id += 1
        if 9 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + \
                            self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1
        if 10 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 11 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 12 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 13 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1
        if 14 in self.controllable_param:
            jntlimit = x[cur_id] * (self.ankle_jnt_range[1] - self.ankle_jnt_range[0]) + \
                            self.ankle_jnt_range[0]
            self.simulator.model.jnt_range[-1][0] = -jntlimit
            self.simulator.model.jnt_range[-1][1] = jntlimit
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


##############################################################################################################
################################  CartPole SwingUp #####################################################################
##############################################################################################################
class CartPoleManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.05, 1.0] # mass range
        self.joint_damping = [0.0, 1.0]
        self.actuator_strength = [20, 50]
        self.attach_width = [0.05, 0.3]
        self.cartmass = [0.05, 5.0]
        self.jug_mass = [0.2, 3.0]

        self.activated_param = [0]
        self.controllable_param = [0]
        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_mass = self.simulator.dart_world.skeletons[-1].bodynodes[2].mass()
        mass_param = (cur_mass - self.range[0]) / (self.range[1] - self.range[0])

        cur_damping = self.simulator.robot_skeleton.joints[1].damping_coefficient(0)
        damping_param = (cur_damping - self.joint_damping[0]) / (self.joint_damping[1] - self.joint_damping[0])

        cur_strength = self.simulator.action_scale[0]
        strength_param = (cur_strength - self.actuator_strength[0]) / (self.actuator_strength[1] - self.actuator_strength[0])

        width = self.simulator.robot_skeleton.bodynodes[-1].shapenodes[0].shape.size()[0]
        width_param = (width - self.attach_width[0]) / (self.attach_width[1] - self.attach_width[0])

        cur_cartmass = self.simulator.dart_world.skeletons[-1].bodynodes[0].mass()
        cartmass_param = (cur_cartmass - self.cartmass[0]) / (self.cartmass[1] - self.cartmass[0])

        jug_mass = self.simulator.dart_world.skeletons[1].bodynodes[0].mass()
        jug_mass_param = (jug_mass - self.jug_mass[0]) / (self.jug_mass[1] - self.jug_mass[0])

        return np.array([mass_param, damping_param, strength_param, width_param, cartmass_param, jug_mass_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            mass = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.dart_world.skeletons[-1].bodynodes[2].set_mass(mass)
            cur_id += 1
        if 1 in self.controllable_param:
            damping = x[cur_id] * (self.joint_damping[1] - self.joint_damping[0]) + self.joint_damping[0]
            self.simulator.robot_skeleton.joints[1].set_damping_coefficient(0, damping)
            cur_id += 1
        if 2 in self.controllable_param:
            strength = x[cur_id] * (self.actuator_strength[1] - self.actuator_strength[0]) + self.actuator_strength[0]
            self.simulator.action_scale[0] = strength
            cur_id += 1
        if 3 in self.controllable_param:
            width = x[cur_id] * (self.attach_width[1] - self.attach_width[0]) + self.attach_width[0]
            size = np.copy(self.simulator.robot_skeleton.bodynodes[-1].shapenodes[0].shape.size())
            size[0] = width
            for i in range(len(self.simulator.robot_skeleton.bodynodes[-1].shapenodes)):
                self.simulator.robot_skeleton.bodynodes[-1].shapenodes[i].shape.set_size(size)
            size = np.copy(size ** 2)
            mass = self.simulator.dart_world.skeletons[-1].bodynodes[2].mass()
            self.simulator.robot_skeleton.bodynodes[-1].set_inertia_entries(1.0/12*mass*(size[1]+size[2]), 1.0/12*mass*(size[0]+size[2]), 1.0/12*mass*(size[1]+size[0]))
            cur_id += 1
        if 4 in self.controllable_param:
            mass = x[cur_id] * (self.cartmass[1] - self.cartmass[0]) + self.cartmass[0]
            self.simulator.dart_world.skeletons[-1].bodynodes[0].set_mass(mass)
            cur_id += 1
        if 5 in self.controllable_param:
            jug_mass = x[cur_id] * (self.jug_mass[1] - self.jug_mass[0]) + self.jug_mass[0]
            self.simulator.dart_world.skeletons[1].bodynodes[0].set_mass(jug_mass)
            self.simulator.dart_world.skeletons[2].bodynodes[0].set_mass(jug_mass)
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(0.0, 1.0, len(self.activated_param))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class walker2dParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mass_range = [2.0, 10.0]
        self.damping_range = [0.01, 0.2]
        self.friction_range = [0.2, 1.0] # friction range
        self.restitution_range = [0.0, 0.8]
        self.power_range = [50, 150]
        self.ankle_power_range = [10, 50]
        self.frame_skip_range = [4, 10]
        self.up_noise_range = [0.0, 1.0]
        self.left_leg_power = [20, 150]
        self.right_leg_power = [20, 150]

        self.activated_param = [0, 3]#[0,1,2,3,4,5,6,   7,8,9,10,11,12,13,14]#[0,1,2,3,4,5, 17, 19, 20]#[0,1,2,3,4,5,6,  7,8,9,10,11,12,  13, 14, 15, 16]
        self.controllable_param = [0, 3]#[0,1,2,3,4,5,6,   7,8,9,10,11,12,13,14]#[0,1,2,3,4,5, 17, 19, 20]#[0,1,2,3,4,5,6,  7,8,9,10,11,12,  13, 14, 15, 16]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        mass_param = []
        for bid in range(2, 9):
            cur_mass = self.simulator.robot_skeleton.bodynodes[bid].m
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 9):
            cur_damp = self.simulator.robot_skeleton.joints[jid].damping_coefficient(0)
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))\

        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.friction_range[0]) / (self.friction_range[1] - self.friction_range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_power = self.simulator.action_scale[0]
        power_param = (cur_power - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_ankl_power = self.simulator.action_scale[2]
        ank_power_param = (cur_ankl_power - self.ankle_power_range[0]) / (self.ankle_power_range[1] - self.ankle_power_range[0])

        cur_frameskip = self.simulator.frame_skip
        frameskip_param = (cur_frameskip - self.frame_skip_range[0]) / (
                self.frame_skip_range[1] - self.frame_skip_range[0])

        cur_up_noise = self.simulator.UP_noise_level
        up_noise_param = (cur_up_noise - self.up_noise_range[0]) / (self.up_noise_range[1] - self.up_noise_range[0])

        cur_left_leg_power = self.simulator.action_scale[0]
        ll_power_param = (cur_left_leg_power - self.left_leg_power[0]) / (
                    self.left_leg_power[1] - self.left_leg_power[0])

        cur_right_leg_power = self.simulator.action_scale[3]
        rl_power_param = (cur_right_leg_power - self.right_leg_power[0]) / (
                self.right_leg_power[1] - self.right_leg_power[0])

        return np.array(mass_param+damp_param+[friction_param, restitution_param, power_param, ank_power_param,
                                               frameskip_param,  up_noise_param, ll_power_param,
                                               rl_power_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0

        for bid in range(0, 7):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.robot_skeleton.bodynodes[bid+2].set_mass(mass)
                cur_id += 1
        for jid in range(7, 13):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.robot_skeleton.joints[jid - 4].set_damping_coefficient(0, damp)
                cur_id += 1

        if 13 in self.controllable_param:
            friction = x[cur_id] * (self.friction_range[1] - self.friction_range[0]) + self.friction_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 14 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        if 15 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.action_scale[[0,1,3,4]] = power
            cur_id += 1
        if 16 in self.controllable_param:
            ank_power = x[cur_id] * (self.ankle_power_range[1] - self.ankle_power_range[0]) + self.ankle_power_range[0]
            self.simulator.action_scale[[2,5]] = ank_power
            cur_id += 1
        if 17 in self.controllable_param:
            frame_skip = x[cur_id] * (self.frame_skip_range[1] - self.frame_skip_range[0]) + self.frame_skip_range[0]
            self.simulator.frame_skip = int(frame_skip)
            cur_id += 1
        if 18 in self.controllable_param:
            up_noise = x[cur_id] * (self.up_noise_range[1] - self.up_noise_range[0]) + self.up_noise_range[0]
            self.simulator.UP_noise_level = up_noise
            cur_id += 1
        if 19 in self.controllable_param:
            ll_power = x[cur_id] * (self.left_leg_power[1] - self.left_leg_power[0]) + self.left_leg_power[0]
            self.simulator.action_scale[[0, 1, 2]] = ll_power
            cur_id += 1
        if 20 in self.controllable_param:
            rl_power = x[cur_id] * (self.right_leg_power[1] - self.right_leg_power[0]) + self.right_leg_power[0]
            self.simulator.action_scale[[3, 4, 5]] = rl_power
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class mjWalkerParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mass_range = [2.0, 15.0]
        self.range = [0.5, 2.0]  # friction range
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.tilt_z_range = [-0.18, 0.18]

        self.activated_param = [6,7]
        self.controllable_param = [6,7]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        mass_param = []
        for bid in range(1, 8):
            cur_mass = self.simulator.model.body_mass[bid]
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        cur_tiltz = self.simulator.tilt_z
        tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])

        params = np.array(mass_param + [friction_param, rest_param ,solimp_param, solref_param, armature_param, tiltz_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        for bid in range(0, 7):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.model.body_mass[bid] = mass
                cur_id += 1

        if 7 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            for bn in range(len(self.simulator.model.geom_friction)):
                self.simulator.model.geom_friction[bn][0] = friction
            cur_id += 1
        if 8 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 9 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 10 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 11 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 9):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1
        if 12 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.simulator.tilt_z = tiltz
            self.simulator.model.opt.gravity[:] = [9.81 * np.sin(tiltz), 0.0, -9.81 * np.cos(tiltz)]
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


class antParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mass_range = [1.0, 5.0]
        self.damping_range = [0.2, 2.0]
        self.friction_range = [0.2, 1.0]  # friction range
        self.restitution_range = [0.0, 0.5]
        self.power_range = [50, 200]
        self.tilt_z_range = [-0.78, 0.78]
        self.tilt_x_range = [-0.78, 0.78]

        self.activated_param = [0,3,6,9,12,21,22,23]
        self.controllable_param = [0,3,6,9,12,21,22,23]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        mass_param = []
        for bid in range(1, 14):
            cur_mass = self.simulator.robot_skeleton.bodynodes[bid].m
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in [3,4, 6,7, 9,10, 12,13]:
            cur_damp = self.simulator.robot_skeleton.joints[jid].damping_coefficient(0)
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))\

        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.friction_range[0]) / (self.friction_range[1] - self.friction_range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_power = self.simulator.action_scale
        power_param = (cur_power - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_tiltx = self.simulator.tilt_x
        tiltx_param = (cur_tiltx - self.tilt_x_range[0]) / (self.tilt_x_range[1] - self.tilt_x_range[0])

        cur_tiltz = self.simulator.tilt_z
        tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])

        return np.array(mass_param+damp_param+[friction_param, restitution_param, power_param, tiltx_param, tiltz_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0

        for bid in range(0, 13):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.robot_skeleton.bodynodes[bid+1].set_mass(mass)
                cur_id += 1
        for id, jid in enumerate([14,15, 17,18, 20,21, 23,24]):
            if id+13 in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.robot_skeleton.joints[jid - 11].set_damping_coefficient(0, damp)
                cur_id += 1

        if 21 in self.controllable_param:
            friction = x[cur_id] * (self.friction_range[1] - self.friction_range[0]) + self.friction_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 22 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        if 23 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.action_scale = power
            cur_id += 1

        if 24 in self.controllable_param:
            tiltx = x[cur_id] * (self.tilt_x_range[1] - self.tilt_x_range[0]) + self.tilt_x_range[0]
            self.simulator.tilt_x = tiltx
            self.simulator.dart_world.set_gravity([0.0, -9.81 * np.cos(tiltx), 9.81*np.sin(tiltx)])
            cur_id += 1
        if 25 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.simulator.tilt_z = tiltz
            self.simulator.dart_world.set_gravity([9.81 * np.sin(tiltz), -9.81 * np.cos(tiltz), 0.0])
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


class cheetahParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mass_range = [0.6, 1.5]
        self.damping_range = [0.7, 1.3]
        self.stiff_range = [0.7, 1.3]
        self.friction_range = [0.4, 1.5]  # friction range
        self.restitution_range = [0.0, 2.0]
        self.gact_scale_range = [0.6, 1.5]
        self.tilt_z_range = [-0.78, 0.78]

        self.activated_param = [0, 1, 2, 3, 4, 5, 6,7, 8,9,10,11,12,13, 14,15,16,17,18,19, 20,21,22]
        self.controllable_param = [0, 1, 2, 3, 4, 5, 6,7, 8,9,10,11,12,13, 14,15,16,17,18,19, 20,21,22]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        mass_param = []
        for bid in range(2, 10):
            cur_mass_ratio = self.simulator.robot_skeleton.bodynodes[bid].m / self.simulator.default_mass[bid]
            mass_param.append((cur_mass_ratio - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in [4,5,6,7,8,9]:
            cur_damp_ratio = self.simulator.robot_skeleton.joints[jid].damping_coefficient(0) / self.simulator.default_dampings[jid]
            damp_param.append((cur_damp_ratio - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))

        stiff_param = []
        for jid in [4, 5, 6, 7, 8, 9]:
            cur_stiff_ratio = self.simulator.robot_skeleton.joints[jid].spring_stiffness(0) / self.simulator.default_stiffness[jid]
            stiff_param.append((cur_stiff_ratio - self.stiff_range[0]) / (self.stiff_range[1] - self.stiff_range[0]))

        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.friction_range[0]) / (self.friction_range[1] - self.friction_range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_power = self.simulator.g_action_scaler
        power_param = (cur_power - self.gact_scale_range[0]) / (self.gact_scale_range[1] - self.gact_scale_range[0])

        cur_tiltz = self.simulator.tilt_z
        tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])


        return np.array(mass_param+damp_param+stiff_param+[friction_param, restitution_param, power_param, tiltz_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0

        for bid in range(0, 8):
            if bid in self.controllable_param:
                mass_ratio = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                mass = mass_ratio * self.simulator.default_mass[bid+2]
                self.simulator.robot_skeleton.bodynodes[bid+2].set_mass(mass)
                cur_id += 1
        for id, jid in enumerate([4,5,6,7,8,9]):
            if id+8 in self.controllable_param:
                damp_ratio = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                damp = damp_ratio * self.simulator.default_dampings[jid]
                self.simulator.robot_skeleton.joints[jid].set_damping_coefficient(0, damp)
                cur_id += 1

        for id, jid in enumerate([4,5,6,7,8,9]):
            if id+14 in self.controllable_param:
                stiff_ratio = x[cur_id] * (self.stiff_range[1] - self.stiff_range[0]) + self.stiff_range[0]
                stiff = stiff_ratio * self.simulator.default_stiffness[jid]
                self.simulator.robot_skeleton.joints[jid].set_spring_stiffness(0, stiff)
                cur_id += 1

        if 20 in self.controllable_param:
            friction = x[cur_id] * (self.friction_range[1] - self.friction_range[0]) + self.friction_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 21 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        if 22 in self.controllable_param:
            power = x[cur_id] * (self.gact_scale_range[1] - self.gact_scale_range[0]) + self.gact_scale_range[0]
            self.simulator.g_action_scaler = power
            cur_id += 1

        if 23 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.simulator.tilt_z = tiltz
            self.simulator.dart_world.set_gravity([9.81 * np.sin(tiltz), -9.81 * np.cos(tiltz), 0.0])
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


class mjcheetahParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.2, 2.0]  # friction range
        self.velrew_weight_range = [-1.0, 1.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.tilt_z_range = [-0.18, 0.18]

        self.activated_param = [2]
        self.controllable_param = [2]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (
                self.velrew_weight_range[1] - self.velrew_weight_range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        cur_tiltz = self.simulator.tilt_z
        tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])

        params = np.array([friction_param, velrew_param, rest_param ,solimp_param, solref_param, armature_param, tiltz_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.model.geom_friction[-1][0] = friction
            cur_id += 1
        if 1 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + \
                            self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1
        if 2 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 3 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 4 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 5 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1
        if 6 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.simulator.tilt_z = tiltz
            self.simulator.model.opt.gravity[:] = [9.81 * np.sin(tiltz), 0.0, -9.81 * np.cos(tiltz)]
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class darwinSquatParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mass_rat_range = [0.8, 1.2]
        self.imu_offset = [-0.02, 0.02]  # in cm, 3 dim
        self.kp_rat_range = [-0.3, 0.3]
        self.kd_rat_range = [-0.3, 0.3]

        self.kp_range = [0.0, 100]
        self.kd_range = [0.0, 1.0]
        self.joint_damping_range = [0.0, 1.0]
        self.joint_friction_range = [0.0, 0.3]

        self.com_offset_range = [-0.03, 0.03]  # index 13, denotes MP_BODY com offset in x direction for now

        self.vel_lim_range = [1.0, 10.0]

        self.ground_friction_range = [0.2, 3.0]

        self.activated_param = [8, 9, 10, 11, 13]
        self.controllable_param = [8, 9, 10, 11, 13]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_mass_rat = self.simulator.mass_ratio
        mass_param = (cur_mass_rat - self.mass_rat_range[0]) / (self.mass_rat_range[1] - self.mass_rat_range[0])

        cur_imu_x = self.simulator.imu_offset_deviation[0]
        imu_x_param = (cur_imu_x - self.imu_offset[0]) / (self.imu_offset[1] - self.imu_offset[0])

        cur_imu_y = self.simulator.imu_offset_deviation[1]
        imu_y_param = (cur_imu_y - self.imu_offset[0]) / (self.imu_offset[1] - self.imu_offset[0])

        cur_imu_z = self.simulator.imu_offset_deviation[2]
        imu_z_param = (cur_imu_z - self.imu_offset[0]) / (self.imu_offset[1] - self.imu_offset[0])

        if self.simulator.kp is not None:
            cur_kp = self.simulator.kp
            kp_param = (cur_kp - self.kp_range[0]) / (self.kp_range[1] - self.kp_range[0])
            cur_kd = self.simulator.kd
            kd_param = (cur_kd - self.kd_range[0]) / (self.kd_range[1] - self.kd_range[0])
        else:
            kp_param = 0.0
            kd_param = 0.0
        cur_damping = self.simulator.robot_skeleton.dofs[-1].damping_coefficient()
        damping_param = (cur_damping - self.joint_damping_range[0]) / (self.joint_damping_range[1] - self.joint_damping_range[0])
        cur_jt_friction = self.simulator.robot_skeleton.dofs[-1].coulomb_friction()
        jt_friction_param = (cur_jt_friction - self.joint_friction_range[0]) / (self.joint_friction_range[1] - self.joint_friction_range[0])

        cur_hip_p_ratio = self.simulator.kp_ratios[2] - self.simulator.default_kp_ratios[2]
        hip_p_ratio_param = (cur_hip_p_ratio - self.kp_rat_range[0]) / (self.kp_rat_range[1] - self.kp_rat_range[0])
        cur_knee_p_ratio = self.simulator.kp_ratios[3] - self.simulator.default_kp_ratios[3]
        knee_p_ratio_param = (cur_knee_p_ratio - self.kp_rat_range[0]) / (self.kp_rat_range[1] - self.kp_rat_range[0])
        cur_ankle_p_ratio = self.simulator.kp_ratios[4] - self.simulator.default_kp_ratios[4]
        ankle_p_ratio_param = (cur_ankle_p_ratio - self.kp_rat_range[0]) / (self.kp_rat_range[1] - self.kp_rat_range[0])

        cur_body_com_x = self.simulator.robot_skeleton.bodynodes[1].local_com()[0] - self.simulator.initial_local_coms[1][0]
        body_com_x_param = (cur_body_com_x - self.com_offset_range[0]) / (self.com_offset_range[1] - self.com_offset_range[0])

        vel_lim_param = (self.simulator.joint_vel_limit - self.vel_lim_range[0]) / (self.vel_lim_range[1] - self.vel_lim_range[0])

        fric_param = (self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff() - self.ground_friction_range[0]) / (
                self.ground_friction_range[1] - self.ground_friction_range[0])

        cur_hip_d_ratio = self.simulator.kd_ratios[2] - self.simulator.default_kd_ratios[2]
        hip_d_ratio_param = (cur_hip_d_ratio - self.kd_rat_range[0]) / (self.kd_rat_range[1] - self.kd_rat_range[0])
        cur_knee_d_ratio = self.simulator.kd_ratios[3] - self.simulator.default_kd_ratios[3]
        knee_d_ratio_param = (cur_knee_d_ratio - self.kd_rat_range[0]) / (self.kd_rat_range[1] - self.kd_rat_range[0])
        cur_ankle_d_ratio = self.simulator.kd_ratios[4] - self.simulator.default_kd_ratios[4]
        ankle_d_ratio_param = (cur_ankle_d_ratio - self.kd_rat_range[0]) / (self.kd_rat_range[1] - self.kd_rat_range[0])

        return np.array([mass_param, imu_x_param, imu_y_param, imu_z_param,
                         kp_param, kd_param, damping_param, jt_friction_param,
                         hip_p_ratio_param, knee_p_ratio_param, ankle_p_ratio_param, body_com_x_param, vel_lim_param,
                         fric_param, hip_d_ratio_param, knee_d_ratio_param, ankle_d_ratio_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0

        if 0 in self.controllable_param:
            mass_rat = x[cur_id] * (self.mass_rat_range[1] - self.mass_rat_range[0]) + self.mass_rat_range[0]
            self.simulator.mass_ratio = mass_rat
            cur_id += 1
        if 1 in self.controllable_param:
            imu_x = x[cur_id] * (self.imu_offset[1] - self.imu_offset[0]) + self.imu_offset[0]
            self.simulator.imu_offset_deviation[0] = imu_x
            cur_id += 1
        if 2 in self.controllable_param:
            imu_y = x[cur_id] * (self.imu_offset[1] - self.imu_offset[0]) + self.imu_offset[0]
            self.simulator.imu_offset_deviation[1] = imu_y
            cur_id += 1
        if 3 in self.controllable_param:
            imu_z = x[cur_id] * (self.imu_offset[1] - self.imu_offset[0]) + self.imu_offset[0]
            self.simulator.imu_offset_deviation[2] = imu_z
            cur_id += 1

        if 4 in self.controllable_param:
            kp = x[cur_id] * (self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
            self.simulator.kp = kp
            cur_id += 1
        if 5 in self.controllable_param:
            kd = x[cur_id] * (self.kd_range[1] - self.kd_range[0]) + self.kd_range[0]
            self.simulator.kd = kd
            cur_id += 1
        if 6 in self.controllable_param:
            damping = x[cur_id] * (self.joint_damping_range[1] - self.joint_damping_range[0]) + self.joint_damping_range[0]
            for i in range(6, self.simulator.robot_skeleton.num_dofs()):
                self.simulator.robot_skeleton.dofs[i].set_damping_coefficient(damping)
            cur_id += 1
        if 7 in self.controllable_param:
            jt_friction = x[cur_id] * (self.joint_friction_range[1] - self.joint_friction_range[0]) + self.joint_friction_range[0]
            for i in range(6, self.simulator.robot_skeleton.num_dofs()):
                self.simulator.robot_skeleton.dofs[i].set_coulomb_friction(jt_friction)
            cur_id += 1

        if 8 in self.controllable_param:
            kp_rat = x[cur_id] * (self.kp_rat_range[1] - self.kp_rat_range[0]) + self.kp_rat_range[0] + self.simulator.default_kp_ratios[2]
            self.simulator.kp_ratios[2] = kp_rat
            cur_id += 1
        if 9 in self.controllable_param:
            kp_rat = x[cur_id] * (self.kp_rat_range[1] - self.kp_rat_range[0]) + self.kp_rat_range[0] + self.simulator.default_kp_ratios[3]
            self.simulator.kp_ratios[3] = kp_rat
            cur_id += 1
        if 10 in self.controllable_param:
            kp_rat = x[cur_id] * (self.kp_rat_range[1] - self.kp_rat_range[0]) + self.kp_rat_range[0] + self.simulator.default_kp_ratios[4]
            self.simulator.kp_ratios[4] = kp_rat
            cur_id += 1

        if 11 in self.controllable_param:
            com = x[cur_id] * (self.com_offset_range[1] - self.com_offset_range[0]) + self.com_offset_range[0]
            init_com = np.copy(self.simulator.initial_local_coms[1])
            init_com[0] += com
            self.simulator.robot_skeleton.bodynodes[1].set_local_com(init_com)
            cur_id += 1

        if 12 in self.controllable_param:
            vel_lim = x[cur_id] * (self.vel_lim_range[1] - self.vel_lim_range[0]) + self.vel_lim_range[0]
            self.simulator.joint_vel_limit = vel_lim
            cur_id += 1

        if 13 in self.controllable_param:
            fric = x[cur_id] * (self.ground_friction_range[1] - self.ground_friction_range[0]) + self.ground_friction_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(fric)
            for bn in self.simulator.robot_skeleton.bodynodes:
                bn.set_friction_coeff(fric)
            cur_id += 1

        if 14 in self.controllable_param:
            kd_rat = x[cur_id] * (self.kd_rat_range[1] - self.kd_rat_range[0]) + self.kd_rat_range[0] + self.simulator.default_kd_ratios[2]
            self.simulator.kd_ratios[2] = kd_rat
            cur_id += 1
        if 15 in self.controllable_param:
            kd_rat = x[cur_id] * (self.kd_rat_range[1] - self.kd_rat_range[0]) + self.kd_rat_range[0] + self.simulator.default_kd_ratios[3]
            self.simulator.kd_ratios[3] = kd_rat
            cur_id += 1
        if 16 in self.controllable_param:
            kd_rat = x[cur_id] * (self.kd_rat_range[1] - self.kd_rat_range[0]) + self.kd_rat_range[0] + self.simulator.default_kd_ratios[4]
            self.simulator.kd_ratios[4] = kd_rat
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

# new formatted manager
class darwinParamManager:
    VARIATIONS = 'KP KD KC KP_RATIO KD_RATIO NEURAL_MOTOR VEL_LIM GROUP_JOINT_DAMPING JOINT_DAMPING JOINT_FRICTION TORQUE_LIM COM_OFFSET GROUND_FRICTION BODY_MASSES MASS_RATIO INERTIA_RATIO BODY_COM_OFFSETS KP_RATIO_NEW KD_RATIO_NEW'.split(' ')
    KP, KD, KC, KP_RATIO, KD_RATIO, NEURAL_MOTOR, VEL_LIM, GROUP_JOINT_DAMPING, JOINT_DAMPING, JOINT_FRICTION, \
    TORQUE_LIM, COM_OFFSET, GROUND_FRICTION, BODY_MASSES, MASS_RATIO, INERTIA_RATIO, BODY_COM_OFFSETS, KP_RATIO_NEW, KD_RATIO_NEW = list(
        range(len(VARIATIONS)))
    MU_DIMS = np.array([5, 5, 5, 5, 5, 27, 1, 5, 1, 1, 1, 2, 1, 14, 26, 26, 52, 7, 7])
    MU_UP_BOUNDS = [[200, 200, 200, 200, 200], [1, 1, 1, 1, 1], [10, 10, 10, 10, 10], [3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0], [1] * 27, [15], [1, 1, 1, 1, 1], [1], [1], [20.0], [0.1, 0.05], [3.0],
                    [1.5,  0.1, 0.1, 0.5, 0.1, 0.1,  0.1, 0.5, 0.5, 0.5, 0.1, 0.1, 0.5, 0.1], [1.3]*26, [1.6]*26,
                    [0.05]*52, [5.0]*7, [5.0]*7]
    MU_LOW_BOUNDS = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1] * 27,
                     [2.0], [0, 0, 0, 0, 0], [0], [0], [3.0], [-0.1, -0.05], [0.0],
                     [0.5, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0], [0.7]*26, [0.6]*26,
                     [-0.05]*52, [0.0]*7, [0.0]*7]
    activated_param = np.arange(14)
    #controllable_param = [KP_RATIO, KD_RATIO, NEURAL_MOTOR, VEL_LIM, JOINT_DAMPING, TORQUE_LIM, COM_OFFSET, GROUND_FRICTION]
    # controllable_param = [NEURAL_MOTOR, TORQUE_LIM, COM_OFFSET, GROUND_FRICTION, KP_RATIO_NEW, KD_RATIO_NEW]#, MASS_RATIO, INERTIA_RATIO]
    controllable_param = [KP_RATIO, KD_RATIO, NEURAL_MOTOR, TORQUE_LIM, COM_OFFSET, GROUND_FRICTION]
    MU_UNSCALED = None  # unscaled version of mu

    def __init__(self, simulator):
        self.simulator = simulator
        self.reward_predictor = None

    @property
    def param_dim(self):
        return len(self.activated_param)

    def set_bounds(self, up, lb): # set bounds from sysid results
        current_id = 0
        unnormed_up = self.UnNormalizeMu(up)
        unnormed_lb = self.UnNormalizeMu(lb)
        for mu in self.controllable_param:
            if current_id >= len(up):
                break
            self.MU_UP_BOUNDS[mu] = unnormed_up[current_id:current_id + self.MU_DIMS[mu]]
            self.MU_LOW_BOUNDS[mu] = unnormed_lb[current_id:current_id + self.MU_DIMS[mu]]
            current_id += self.MU_DIMS[mu]
        print('New bounds: ')
        print([[self.VARIATIONS[mu], [self.MU_UP_BOUNDS[mu], self.MU_LOW_BOUNDS[mu]]] for mu in self.controllable_param])

    def NormalizeMu(self, unnorm_mu):
        current_id = 0
        scaled_mu = np.zeros(len(unnorm_mu))
        unnorm_mu = np.array(unnorm_mu)
        for mu in self.controllable_param:
            scaled_mu[current_id:current_id + self.MU_DIMS[mu]] = \
                (unnorm_mu[current_id:current_id + self.MU_DIMS[mu]] - np.array(self.MU_LOW_BOUNDS[mu])) / \
                (np.array(self.MU_UP_BOUNDS[mu]) - np.array(self.MU_LOW_BOUNDS[mu]))
            current_id += self.MU_DIMS[mu]
        return np.copy(scaled_mu)

    def UnNormalizeMu(self, norm_mu):
        unnorm_mu = np.zeros(len(norm_mu))
        current_id = 0
        for mu in self.controllable_param:
            unnorm_mu[current_id:current_id + self.MU_DIMS[mu]] = \
                np.array(norm_mu[current_id:current_id + self.MU_DIMS[mu]]) * \
                (np.array(self.MU_UP_BOUNDS[mu]) - np.array(self.MU_LOW_BOUNDS[mu])) + \
                np.array(self.MU_LOW_BOUNDS[mu])
            current_id += self.MU_DIMS[mu]
        return np.copy(unnorm_mu)

    def get_simulator_parameters(self):
        self.MU_UNSCALED = []

        if self.KP in self.controllable_param:
            self.MU_UNSCALED.append(self.simulator.kp[0]) # arm
            self.MU_UNSCALED.append(self.simulator.kp[6])  # head
            self.MU_UNSCALED.append(self.simulator.kp[8])  # hip
            self.MU_UNSCALED.append(self.simulator.kp[11])  # knee
            self.MU_UNSCALED.append(self.simulator.kp[12])  # head


        if self.KD in self.controllable_param:
            self.MU_UNSCALED.append(self.simulator.kd[0])  # arm
            self.MU_UNSCALED.append(self.simulator.kd[6])  # head
            self.MU_UNSCALED.append(self.simulator.kd[8])  # hip
            self.MU_UNSCALED.append(self.simulator.kd[11])  # knee
            self.MU_UNSCALED.append(self.simulator.kd[12])  # head

        if self.KC in self.controllable_param:
            self.MU_UNSCALED.append(self.simulator.kc[0])  # arm
            self.MU_UNSCALED.append(self.simulator.kc[6])  # head
            self.MU_UNSCALED.append(self.simulator.kc[8])  # hip
            self.MU_UNSCALED.append(self.simulator.kc[11])  # knee
            self.MU_UNSCALED.append(self.simulator.kc[12])  # head

        if self.KP_RATIO in self.controllable_param:
            for rat in self.simulator.kp_ratios:
                self.MU_UNSCALED.append(rat)

        if self.KD_RATIO in self.controllable_param:
            for rat in self.simulator.kd_ratios:
                self.MU_UNSCALED.append(rat)

        if self.NEURAL_MOTOR in self.controllable_param:
            for pm in range(len(self.simulator.NN_motor_parameters)):
                dim = np.prod(self.simulator.NN_motor_parameters[pm].shape)
                self.MU_UNSCALED += np.reshape(self.simulator.NN_motor_parameters[pm], (dim,)).tolist()

        if self.VEL_LIM in self.controllable_param:
            self.MU_UNSCALED.append(self.simulator.joint_vel_limit)

        if self.GROUP_JOINT_DAMPING in self.controllable_param:
            self.MU_UNSCALED.append(self.simulator.actuated_dofs[0].damping_coefficient())  # arm
            self.MU_UNSCALED.append(self.simulator.actuated_dofs[6].damping_coefficient())  # head
            self.MU_UNSCALED.append(self.simulator.actuated_dofs[8].damping_coefficient())  # hip
            self.MU_UNSCALED.append(self.simulator.actuated_dofs[11].damping_coefficient())  # knee
            self.MU_UNSCALED.append(self.simulator.actuated_dofs[12].damping_coefficient())  # head


        if self.JOINT_DAMPING in self.controllable_param:
            self.MU_UNSCALED.append(self.simulator.actuated_dofs[0].damping_coefficient())

        if self.JOINT_FRICTION in self.controllable_param:
            self.MU_UNSCALED.append(self.simulator.actuated_dofs[6].coulomb_friction())

        if self.TORQUE_LIM in self.controllable_param:
            self.MU_UNSCALED.append(self.simulator.torqueLimits)

        if self.COM_OFFSET in self.controllable_param:
            init_com = np.copy(self.simulator.initial_local_coms[0])
            self.MU_UNSCALED.append(self.simulator.body_parts[0].local_com()[0] - init_com[0])
            self.MU_UNSCALED.append(self.simulator.body_parts[0].local_com()[2] - init_com[2])

        if self.GROUND_FRICTION in self.controllable_param:
            self.MU_UNSCALED.append(self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff())

        if self.MASS_RATIO in self.controllable_param:
            self.MU_UNSCALED += self.simulator.mass_ratios.tolist()

        if self.INERTIA_RATIO in self.controllable_param:
            self.MU_UNSCALED += self.simulator.inertia_ratios.tolist()

        if self.BODY_COM_OFFSETS in self.controllable_param:
            for bid in range(0, 26):
                init_com = np.copy(self.simulator.initial_local_coms[bid])
                self.MU_UNSCALED.append(self.simulator.body_parts[bid].local_com()[0] - init_com[0])
                self.MU_UNSCALED.append(self.simulator.body_parts[bid].local_com()[2] - init_com[2])

        if self.KP_RATIO_NEW in self.controllable_param:
            for rat in self.simulator.kp_ratios:
                self.MU_UNSCALED.append(rat)

        if self.KD_RATIO_NEW in self.controllable_param:
            for rat in self.simulator.kd_ratios:
                self.MU_UNSCALED.append(rat)

        return self.NormalizeMu(self.MU_UNSCALED)[self.activated_param]


    def set_simulator_parameters(self, x):
        assert (len(x) == np.sum(self.MU_DIMS[self.controllable_param])), str(len(x))+' '+str(np.sum(self.MU_DIMS[self.controllable_param]))

        self.MU_UNSCALED = self.UnNormalizeMu(x)

        current_id = 0
        if self.KP in self.controllable_param:
            self.simulator.kp = np.zeros(20)
            # arms
            self.simulator.kp[0:6] = self.MU_UNSCALED[current_id]
            # head
            self.simulator.kp[6:8] = self.MU_UNSCALED[current_id + 1]
            # hip
            self.simulator.kp[8:11] = self.MU_UNSCALED[current_id + 2]
            self.simulator.kp[14:17] = self.MU_UNSCALED[current_id + 2]
            # knee
            self.simulator.kp[11] = self.MU_UNSCALED[current_id + 3]
            self.simulator.kp[17] = self.MU_UNSCALED[current_id + 3]
            # ankle
            self.simulator.kp[12:14] = self.MU_UNSCALED[current_id + 4]
            self.simulator.kp[18:20] = self.MU_UNSCALED[current_id + 4]

            current_id += self.MU_DIMS[self.KP]

        if self.KD in self.controllable_param:
            self.simulator.kd = np.zeros(20)
            # arms
            self.simulator.kd[0:6] = self.MU_UNSCALED[current_id]
            # head
            self.simulator.kd[6:8] = self.MU_UNSCALED[current_id + 1]
            # hip
            self.simulator.kd[8:11] = self.MU_UNSCALED[current_id + 2]
            self.simulator.kd[14:17] = self.MU_UNSCALED[current_id + 2]
            # knee
            self.simulator.kd[11] = self.MU_UNSCALED[current_id + 3]
            self.simulator.kd[17] = self.MU_UNSCALED[current_id + 3]
            # ankle
            self.simulator.kd[12:14] = self.MU_UNSCALED[current_id + 4]
            self.simulator.kd[18:20] = self.MU_UNSCALED[current_id + 4]
            current_id += self.MU_DIMS[self.KD]

        if self.KC in self.controllable_param:
            self.simulator.kc = np.zeros(20)
            # arms
            self.simulator.kc[0:6] = self.MU_UNSCALED[current_id]
            # head
            self.simulator.kc[6:8] = self.MU_UNSCALED[current_id + 1]
            # hip
            self.simulator.kc[8:11] = self.MU_UNSCALED[current_id + 2]
            self.simulator.kc[14:17] = self.MU_UNSCALED[current_id + 2]
            # knee
            self.simulator.kc[11] = self.MU_UNSCALED[current_id + 3]
            self.simulator.kc[17] = self.MU_UNSCALED[current_id + 3]
            # ankle
            self.simulator.kc[12:14] = self.MU_UNSCALED[current_id + 4]
            self.simulator.kc[18:20] = self.MU_UNSCALED[current_id + 4]
            current_id += self.MU_DIMS[self.KC]

        if self.KP_RATIO in self.controllable_param:
            self.simulator.kp_ratios = self.MU_UNSCALED[current_id:current_id + self.MU_DIMS[self.KP_RATIO]]
            current_id += self.MU_DIMS[self.KP_RATIO]

        if self.KD_RATIO in self.controllable_param:
            self.simulator.kd_ratios = self.MU_UNSCALED[current_id:current_id + self.MU_DIMS[self.KD_RATIO]]
            current_id += self.MU_DIMS[self.KD_RATIO]

        if self.NEURAL_MOTOR in self.controllable_param:
            self.simulator.NN_motor = True
            for pm in range(len(self.simulator.NN_motor_parameters)):
                dim = np.prod(self.simulator.NN_motor_parameters[pm].shape)
                shape = self.simulator.NN_motor_parameters[pm].shape
                self.simulator.NN_motor_parameters[pm] = np.reshape(self.MU_UNSCALED[current_id: current_id + dim],
                                                                     shape)
                current_id += dim

        if self.VEL_LIM in self.controllable_param:
            self.simulator.joint_vel_limit = self.MU_UNSCALED[current_id]
            current_id += self.MU_DIMS[self.VEL_LIM]

        if self.GROUP_JOINT_DAMPING in self.controllable_param:
            # arms
            for i in range(0, 6):
                j = self.simulator.actuated_dofs[i]
                j.set_damping_coefficient(self.MU_UNSCALED[current_id])
            # head
            for i in range(6, 8):
                j = self.simulator.actuated_dofs[i]
                j.set_damping_coefficient(self.MU_UNSCALED[current_id + 1])
            # hip
            for i in range(8, 11):
                j = self.simulator.actuated_dofs[i]
                j.set_damping_coefficient(self.MU_UNSCALED[current_id + 2])
            for i in range(14, 17):
                j = self.simulator.actuated_dofs[i]
                j.set_damping_coefficient(self.MU_UNSCALED[current_id + 2])
            # knee
            j = self.simulator.actuated_dofs[11]
            j.set_damping_coefficient(self.MU_UNSCALED[current_id + 3])
            j = self.simulator.actuated_dofs[17]
            j.set_damping_coefficient(self.MU_UNSCALED[current_id + 3])
            # ankle
            for i in range(12, 16):
                j = self.simulator.actuated_dofs[i]
                j.set_damping_coefficient(self.MU_UNSCALED[current_id + 4])
            for i in range(18, 20):
                j = self.simulator.actuated_dofs[i]
                j.set_damping_coefficient(self.MU_UNSCALED[current_id + 4])

            current_id += self.MU_DIMS[self.GROUP_JOINT_DAMPING]

        if self.JOINT_DAMPING in self.controllable_param:
            joint_damping = self.MU_UNSCALED[current_id]
            for i in range(len(self.simulator.actuated_dofs)):
                j = self.simulator.actuated_dofs[i]
                j.set_damping_coefficient(joint_damping)

            current_id += self.MU_DIMS[self.JOINT_DAMPING]

        if self.JOINT_FRICTION in self.controllable_param:
            joint_friction = self.MU_UNSCALED[current_id]
            for i in range(len(self.simulator.actuated_dofs)):
                j = self.simulator.actuated_dofs[i]
                j.set_coulomb_friction(joint_friction)
            current_id += self.MU_DIMS[self.JOINT_FRICTION]

        if self.TORQUE_LIM in self.controllable_param:
            self.simulator.torqueLimits = self.MU_UNSCALED[current_id]
            current_id += self.MU_DIMS[self.TORQUE_LIM]

        if self.COM_OFFSET in self.controllable_param:
            init_com = np.copy(self.simulator.initial_local_coms[0])
            init_com[0] += self.MU_UNSCALED[current_id]
            init_com[2] += self.MU_UNSCALED[current_id + 1]
            self.simulator.body_parts[0].set_local_com(init_com)
            current_id += self.MU_DIMS[self.COM_OFFSET]

        if self.GROUND_FRICTION in self.controllable_param:
            for bn in self.simulator.dart_world.skeletons[0].bodynodes:
                bn.set_friction_coeff(self.MU_UNSCALED[current_id])
            for bn in self.simulator.robot_skeleton.bodynodes:
                bn.set_friction_coeff(self.MU_UNSCALED[current_id])
            current_id += self.MU_DIMS[self.GROUND_FRICTION]

        if self.MASS_RATIO in self.controllable_param:
            self.simulator.mass_ratios = self.MU_UNSCALED[current_id:current_id + self.MU_DIMS[self.MASS_RATIO]]
            current_id += self.MU_DIMS[self.MASS_RATIO]

        if self.INERTIA_RATIO in self.controllable_param:
            self.simulator.inertia_ratios = self.MU_UNSCALED[current_id:current_id + self.MU_DIMS[self.INERTIA_RATIO]]
            current_id += self.MU_DIMS[self.INERTIA_RATIO]

        if self.BODY_COM_OFFSETS in self.controllable_param:
            for bid in range(0, 26):
                init_com = np.copy(self.simulator.initial_local_coms[bid])
                init_com[0] += self.MU_UNSCALED[current_id + bid * 2]
                init_com[2] += self.MU_UNSCALED[current_id + bid * 2 + 1]
                self.simulator.body_parts[bid].set_local_com(init_com)
            current_id += self.MU_DIMS[self.BODY_COM_OFFSETS]

        if self.KP_RATIO_NEW in self.controllable_param:
            self.simulator.kp_ratios = self.MU_UNSCALED[current_id:current_id + self.MU_DIMS[self.KP_RATIO_NEW]]
            current_id += self.MU_DIMS[self.KP_RATIO_NEW]

        if self.KD_RATIO_NEW in self.controllable_param:
            self.simulator.kd_ratios = self.MU_UNSCALED[current_id:current_id + self.MU_DIMS[self.KD_RATIO_NEW]]
            current_id += self.MU_DIMS[self.KD_RATIO_NEW]

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, np.sum(self.MU_DIMS[self.controllable_param]))
        if self.reward_predictor is not None:
            while True:
                pred_rew = self.reward_predictor.predict_reward(x)[0]
                normed_rew = (pred_rew - self.reward_predictor.reward_range[0]) / (self.reward_predictor.reward_range[1] - self.reward_predictor.reward_range[0])
                normed_rew = np.clip(normed_rew, 0.05, 0.95)
                if np.random.random() > normed_rew:
                    break
                else:
                    x = np.random.uniform(-0.05, 1.05, np.sum(self.MU_DIMS[self.controllable_param]))

        self.set_simulator_parameters(x)
