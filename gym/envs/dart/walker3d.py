__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import joblib
import os

from gym.envs.dart.parameter_managers import *
import time
from gym.envs.dart.action_filter import *

class DartWalker3dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*15,[-1.0]*15])
        self.action_scale = np.array([200.0, 200, 200, 250, 60, 80, 100, 60, 60, 250, 60, 80, 100, 60, 60])
        obs_dim = 41

        self.t = 0
        self.target_vel = 2.0
        self.init_tv = 0.0
        self.final_tv = 2.0
        self.tv_endtime = 0.5
        self.smooth_tv_change = True
        self.rand_target_vel = False
        self.init_push = False
        self.enforce_target_vel = True
        self.running_avg_rew_only = True
        self.avg_rew_weighting = []
        self.vel_cache = []
        self.butterworth_filter = True

        self.staged_reward = True

        self.target_ang = None

        self.assist_timeout = 0.0
        self.assist_prob = 1.0 # probability of providing assistance
        self.assist_schedule = [[0.0, [2000, 2000]], [3.0, [1500, 1500]], [6.0, [1125.0, 1125.0]]]

        self.hard_enforce = False
        self.treadmill = False
        self.treadmill_vel = -self.init_tv
        self.treadmill_init_tv = -1.2
        self.treadmill_final_tv = -1.2
        self.treadmill_tv_endtime = 0.04

        self.cur_step = 0
        self.stepwise_rewards = []
        self.conseq_limit_pen = 0 # number of steps lying on the wall
        self.constrain_2d = True
        self.init_balance_pd = 2000.0
        self.init_vel_pd = 2000.0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.energy_weight = 0.1
        self.vel_reward_weight = 3.0

        self.local_spd_curriculum = True
        self.anchor_kp = np.array([0, 0]) * 1.0

        self.learns_turning = False

        # state related
        self.contact_info = np.array([0, 0])
        self.include_additional_info = True
        if self.include_additional_info:
            obs_dim += len(self.contact_info)
        if self.rand_target_vel or self.smooth_tv_change:
            obs_dim += 1

        self.curriculum_id = 0
        self.spd_kp_candidates = None
        #self.param_manager = walker3dManager(self)

        if self.treadmill:
            dart_env.DartEnv.__init__(self, 'walker3d_treadmill.skel', 15, obs_dim, self.control_bounds, disableViewer=True)
        else:
            dart_env.DartEnv.__init__(self, 'walker3d_waist.skel', 15, obs_dim, self.control_bounds,
                                      disableViewer=True, dt=0.002)

        #self.dart_world.set_collision_detector(3) # uncomment if using ODE collision detector
        self.robot_skeleton.set_self_collision_check(True)

        for i in range(1, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(0)

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(5)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(5)

        for bn in self.robot_skeleton.bodynodes:
            if len(bn.shapenodes) > 0:
                shapesize = bn.shapenodes[0].shape.size()
                print('density of ', bn.name, ' is ', bn.mass()/np.prod(shapesize))
        print('Total mass: ', self.robot_skeleton.mass())


        self.obs_perm = np.array([0.0001, -1, 2, -3, -4, -5, -6, 7, 14, -15, -16, 17, 18, -19, 8,
                                                      -9, -10, 11, 12, -13,
                                                      20, 21, -22, 23, -24, -25, -26, -27, 28, 35, -36, -37, 38, 39,
                                                      -40, 29, -30, -31, 32, 33,
                                                      -34, 42, 41, 43])

        self.act_perm = np.array([-0.0001, -1, 2, 9, -10, -11, 12, 13, -14, 3, -4, -5, 6, 7, -8])

        if self.butterworth_filter:
            self.action_filter = ActionFilter(15, 3, int(1.0/self.dt), low_cutoff=0.0, high_cutoff=6.0)

        utils.EzPickle.__init__(self)

    def resample_task(self):
        return [0.0]

    def set_task(self, task_params):
        pass

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


    def do_simulation(self, tau, n_frames):
        provide_assist = np.random.random() < self.assist_prob
        for _ in range(n_frames):
            if self.constrain_2d and self.t < self.assist_timeout and provide_assist:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('h_pelvis'), self.current_pd, 2)
                self.robot_skeleton.bodynode('h_pelvis').add_ext_force(np.array([0, 0, force]))

            if self.enforce_target_vel and not self.hard_enforce and self.t < self.assist_timeout and provide_assist:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('h_pelvis'), self.vel_enforce_kp, 0, self.target_vel)
                self.robot_skeleton.bodynode('h_pelvis').add_ext_force(np.array([force, 0, 0]))

            if self.target_ang is not None:
                tau[4] = self.vel_enforce_kp/100 * (self.target_ang - self.robot_skeleton.q[4])

            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()
            s = self.state_vector()
            if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
                break

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        if self.enforce_target_vel:
            if self.hard_enforce and self.treadmill:
                current_dq_tread = self.dart_world.skeletons[0].dq
                current_dq_tread[0] = self.treadmill_vel# * np.min([self.t/4.0, 1.0])
                self.dart_world.skeletons[0].dq = current_dq_tread
            elif self.hard_enforce:
                current_dq = self.robot_skeleton.dq
                current_dq[0] = self.target_vel
                self.robot_skeleton.dq = current_dq
        self.do_simulation(tau, self.frame_skip)

    def step(self, a):
        if self.butterworth_filter:
            a = self.action_filter.filter_action(a)

        if self.smooth_tv_change:
            self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (self.final_tv - self.init_tv) + self.init_tv
            self.treadmill_vel = (np.min([self.t, self.treadmill_tv_endtime]) / self.treadmill_tv_endtime) * (
                    self.treadmill_final_tv - self.treadmill_init_tv) + self.treadmill_init_tv

        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        if len(self.assist_schedule) > 0:
            for sch in self.assist_schedule:
                if self.t > sch[0]:
                    self.current_pd = sch[1][0]
                    self.vel_enforce_kp = sch[1][1]

        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.bodynodes[0].com()[0]
        self.advance(np.copy(a))

        posafter = self.robot_skeleton.bodynodes[1].com()[0]
        height = self.robot_skeleton.bodynodes[1].com()[1]
        side_deviation = self.robot_skeleton.bodynodes[1].com()[2]
        angle = self.robot_skeleton.q[3]

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([0, 1, 0])) - self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([1, 0, 0])) - self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        self_colliding = False
        self.contact_info = np.array([0, 0])
        l_foot_force = np.array([0.0, 0, 0])
        r_foot_force = np.array([0.0, 0, 0])
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.skel_id1 == contact.skel_id2:
                self_colliding = True
            if contact.skel_id1 + contact.skel_id2 == 1:
                if contact.bodynode1 == self.robot_skeleton.bodynode('h_foot_left') or contact.bodynode2 == \
                        self.robot_skeleton.bodynode('h_foot_left'):
                    self.contact_info[0] = 1
                    l_foot_force += contact.force
                if contact.bodynode1 == self.robot_skeleton.bodynode('h_foot') or contact.bodynode2 == \
                        self.robot_skeleton.bodynode('h_foot'):
                    self.contact_info[1] = 1
                    r_foot_force += contact.force

        alive_bonus = 4.0
        vel = (posafter - posbefore) / self.dt
        self.vel_cache.append(vel)
        self.target_vel_cache.append(self.target_vel)

        if len(self.vel_cache) > int(2.0/self.dt) and (self.running_avg_rew_only):
            self.vel_cache.pop(0)
            self.target_vel_cache.pop(0)

        vel_rew = 0
        if not self.treadmill:
            if self.running_avg_rew_only:
                vel_rew = -self.vel_reward_weight * np.abs(np.mean(self.target_vel_cache) - np.mean(self.vel_cache))
                if self.t < self.tv_endtime:
                    vel_rew *= 0.5
            else:
                vel_rew = -self.vel_reward_weight * np.abs(self.target_vel - vel).sum()
        else:
            if self.running_avg_rew_only:
                append_vel = np.ones(int(1.0/self.dt) - len(self.vel_cache)) * (self.target_vel + self.treadmill_vel)
                vel_rew = -3.0 * (np.abs(self.target_vel + self.treadmill_vel - np.mean(np.append(self.vel_cache, append_vel))))
            else:
                vel_rew = -3.0 * (np.abs(self.target_vel + self.treadmill_vel - vel))
        if self.t < self.tv_endtime:
            vel_rew *= 1.0

        if self.target_ang is not None:
            print(self.robot_skeleton.q[4])
            ang_vel_rew = - 3.0 * np.abs(np.abs(self.target_ang) - np.abs(self.robot_skeleton.q[4]))
        else:
            ang_vel_rew = 0

        action_pen = self.energy_weight * np.abs(a).sum()
        deviation_pen = 3 * abs(side_deviation)

        rot_pen = 1.0 * np.abs(self.robot_skeleton.q[3]) + 0.0 * np.abs(self.robot_skeleton.q[4]) + \
                  1.0 * np.abs(self.robot_skeleton.q[5])

        jump_rew = 10.0 * np.max([(height - 1.3), 0])
        foot_rew = 5*(np.max([self.robot_skeleton.bodynode('h_thigh').C[1], self.robot_skeleton.bodynode('h_thigh_left').C[1]])-0.8)

        reward = vel_rew + alive_bonus - action_pen - deviation_pen - rot_pen# + jump_rew + ang_vel_rew# - contact_pen

        if self.staged_reward:
            reward = 10.0 - action_pen - np.sum(np.abs(self.robot_skeleton.q[3:6])) * 2.0

        pos_rew = alive_bonus - deviation_pen
        neg_pen = vel_rew - action_pen

        self.t += self.dt
        self.cur_step += 1

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > 1.0) and (height < 3.0) and (abs(ang_cos_uwd) < 1.2) and (abs(ang_cos_fwd) < 1.2)
                    and np.abs(angle) < 1.2 and np.abs(self.robot_skeleton.q[5]) < 1.2 and np.abs(side_deviation) < 0.9)

        # if not self.disableViewer:
        #     done = False

        self.stepwise_rewards.append(reward)

        broke_sim = False
        if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
            broke_sim = True

        ob = self._get_obs()
 
        return ob, reward, done, {'pos_rew':pos_rew, 'neg_pen': neg_pen, 'broke_sim':broke_sim, 'pre_state':pre_state,
                                  'vel_rew':vel_rew, 'action_pen':action_pen/self.energy_weight, 'deviation_pen':deviation_pen,
                                  'curriculum_id':self.curriculum_id, 'curriculum_candidates':self.spd_kp_candidates,
                                  'done_return':done, 'dyn_model_id':0, 'state_index':0,
                                  'contact_forces': [l_foot_force, r_foot_force],
                                  'contact_force': l_foot_force + r_foot_force, 'avg_vel':np.mean(self.vel_cache)}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        if self.include_additional_info:
            state = np.concatenate([state, self.contact_info])

        if self.rand_target_vel or self.smooth_tv_change:
            state = np.concatenate([state, [self.target_vel]])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q# + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq# + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)

        if self.rand_target_vel:
            self.target_vel = np.random.uniform(0.8, 2.5)

        if self.local_spd_curriculum:
            self.spd_kp_candidates = [self.anchor_kp]
            self.curriculum_id = np.random.randint(len(self.spd_kp_candidates))
            chosen_curriculum = self.spd_kp_candidates[self.curriculum_id]
            self.init_balance_pd = chosen_curriculum[0]
            self.init_vel_pd = chosen_curriculum[1]

        if self.init_push:
            qpos[0] = self.target_vel
        self.set_state(qpos, qvel)

        if self.staged_reward:
            valid_init = False
            while not valid_init:
                self.set_state(qpos, qvel)
                for i in range(150):
                    self.robot_skeleton.set_forces(np.concatenate([[0, 0, 0, 0, 0, 0], np.random.uniform(-1, 1, len(self.action_scale)) * self.action_scale]))
                    self.dart_world.step()
                s = self.state_vector()
                height = self.robot_skeleton.bodynodes[1].com()[1]
                side_deviation = self.robot_skeleton.bodynodes[1].com()[2]
                angle = self.robot_skeleton.q[3]
                upward = np.array([0, 1, 0])
                upward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([0, 1, 0])) - \
                               self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
                upward_world /= np.linalg.norm(upward_world)
                ang_cos_uwd = np.dot(upward, upward_world)
                ang_cos_uwd = np.arccos(ang_cos_uwd)
                forward = np.array([1, 0, 0])
                forward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([1, 0, 0])) - \
                                self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
                forward_world /= np.linalg.norm(forward_world)
                ang_cos_fwd = np.dot(forward, forward_world)
                ang_cos_fwd = np.arccos(ang_cos_fwd)
                done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                            (height > 1.0) and (height < 3.0) and (abs(ang_cos_uwd) < 1.2) and (abs(ang_cos_fwd) < 1.2)
                            and np.abs(angle) < 1.2 and np.abs(self.robot_skeleton.q[5]) < 1.2 and np.abs(
                            side_deviation) < 0.9)
                if not done:
                    valid_init = True

        self.t = 0
        self.cur_step = 0
        self.stepwise_rewards = []

        self.init_pos = self.robot_skeleton.q[0]

        self.vel_cache = []
        self.target_vel_cache = []

        self.avg_rew_weighting = []

        self.conseq_limit_pen = 0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.contact_info = np.array([0, 0])

        if self.target_ang is not None and np.random.random() < 0.5:
            self.target_ang *= -1

        if self.butterworth_filter:
            self.action_filter.reset_filter(np.zeros(15))

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5


    def resample_reward_function(self):
        # alive bonus, velocity reward, energy penalty
        coef = np.random.random(3)
        self.alive_bonus = coef[0] * 5.0  # 0-5
        self.energy_penalty = coef[1] * 0.01  # 0-0.01
        self.velrew_weight = coef[2] * 5.0 # 0-5
        return coef

    def set_reward_function(self, coef):
        self.alive_bonus = coef[0] * 5.0  # 0-5
        self.energy_penalty = coef[1] * 0.01  # 0-0.01
        self.velrew_weight = coef[2] * 5.0  # 0-5

    def true_reward(self):
        return self.robot_skeleton.q[0]  # position in x direction