# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
import random
import time
import math

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

class DartNoClothHapticTrainer(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.8, -0.6, 0.6])
        self.target2 = np.array([0.8, -0.6, 0.6])
        self.targetActive1 = True
        self.targetActive2 = False
        self.randomActiveTarget = False #if true, 1/4 chance of both active, neither, either 1 or 2
        self.arm = 1 #if 1, left arm (character's perspective), if 2, right

        self.sensorNoise = 0.0
        self.stableReacher = True #if true, no termination reward for touching the target (must hover instead)

        #storage of rewards from previous step for rendering
        self.renderRewards = True
        self.cumulativeReward = 0
        self.pastReward = 0
        self.distreward1 = 0
        self.distreward2 = 0
        self.taureward = 0
        self.dispreward1 = 0
        self.dispreward2 = 0
        self.proxreward1 = 0
        self.proxreward2 = 0
        self.velocityPenalty = 0
        self.prevDq = None

        self.restPoseActive = True
        self.restPoseWeight = 0.05
        self.restPose = np.array([])
        self.restPoseReward = 0
        self.usePoseTarget = False #if true, rest pose is given in policy input

        self.interactiveTarget = False
        
        #22 dof upper body
        self.action_scale = np.ones(11)*10

        self.control_bounds = np.array([np.ones(11), np.ones(11)*-1])
        
        self.reset_number = 0 #debugging
        self.numSteps = 0
        
        self.doROM = False
        self.ROM_period = 200.0
        
        self.targetHistory = []
        self.successHistory = []
        
        #create cloth scene
        clothScene = pyphysx.ClothScene(step=0.01, sheet=True, sheetW=60, sheetH=15, sheetSpacing=0.025)
        
        #intialize the parent env
        observation_size = 66+66 #pose(sin,cos), pose vel, haptics
        if self.targetActive1 is True:
            observation_size += 6
        if self.targetActive2 is True:
            observation_size += 6
        if self.usePoseTarget is True:
            observation_size += 22

        DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='UpperBodyCapsules.skel', frame_skip=4, observation_size=(66+66+6), action_bounds=self.control_bounds)#, disableViewer=True, visualize=False)

        self.restPose = np.array(self.robot_skeleton.q) #by default the rest pose is start pose. Change this in reset if desired.

        for body in self.robot_skeleton.bodynodes:
            print(body.name + ": " + str(body.is_collidable()))

        utils.EzPickle.__init__(self)
        
        self.clothScene.seedRandom(random.randint(1,1000))
        self.clothScene.setFriction(0, 1)
        
        #self.updateClothCollisionStructures(capsules=True, hapticSensors=True)
        
        self.simulateCloth = False
        self.sampleFromHemisphere = False
        
        self.renderDofs = True #if true, show dofs text 
        self.renderForceText = False
        
        self.random_dir = np.array([0,0,1.])
        
        self.tag = "its text"

        for i in range(len(self.robot_skeleton.bodynodes)):
            print(self.robot_skeleton.bodynodes[i])
            
        for i in range(len(self.robot_skeleton.dofs)):
            print(self.robot_skeleton.dofs[i])

    def limits(self, dof_ix):
        return np.array([self.robot_skeleton.dof(dof_ix).position_lower_limit(), self.robot_skeleton.dof(dof_ix).position_upper_limit()])
        
    def _step(self, a):
        #print("step")
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        fingertip = np.array([0.0, -0.06, 0.0])
        wRFingertip1 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        vecR1 = self.target-wRFingertip1
        vecL1 = self.target2-wLFingertip1
        
        #apply action and simulate
        if self.arm == 1:
            tau = np.concatenate([tau, np.zeros(11)])
        else:
            tau = np.concatenate([tau[:3], np.zeros(8), tau[3:], np.zeros(3)])
        self.do_simulation(tau, self.frame_skip)
        
        wRFingertip2 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        wLFingertip2 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        vecR2 = self.target-wRFingertip2
        vecL2 = self.target2-wLFingertip2
        
        #distance to target penalty
        reward_dist1 = 0
        reward_dist2 = 0
        if self.targetActive1:
            reward_dist1 = - np.linalg.norm(vecR2)
        if self.targetActive2:
            reward_dist2 = - np.linalg.norm(vecL2)
        reward_dist = reward_dist1 + reward_dist2
        
        #force magnitude penalty    
        reward_ctrl = - np.linalg.norm(tau) * 0.0001

        self.velocityPenalty = -np.linalg.norm(self.robot_skeleton.dq) *0.001

        accelerationPenalty = 0
        if self.prevDq is not None:
            accelerationPenalty = -np.linalg.norm(self.prevDq - self.robot_skeleton.dq)*0.01
        
        #displacement toward target reward
        reward_progress1 = 0
        reward_progress2 = 0
        if self.targetActive1:
            reward_progress1 += np.dot((wRFingertip2 - wRFingertip1), vecR1/np.linalg.norm(vecR1)) * 100
        if self.targetActive2:
            reward_progress2 += np.dot((wLFingertip2 - wLFingertip1), vecL1/np.linalg.norm(vecL1)) * 100
        reward_progress = reward_progress1 + reward_progress2
        #horizon length penalty
        alive_bonus = -0.001
        
        #proximity to target bonus
        reward_prox1 = 0
        reward_prox2 = 0
        if self.targetActive1 and -reward_dist1 < 0.1:
            reward_prox1 += (0.1+reward_dist1)*40
        if self.targetActive2 and -reward_dist2 < 0.1:
            reward_prox2 += (0.1+reward_dist2)*40
        reward_prox = reward_prox1 + reward_prox2

        #rest pose reward
        self.restPoseReward = 0
        if self.restPoseActive is True and len(self.restPose) == len(self.robot_skeleton.q):
            #print(np.linalg.norm(self.restPose - self.robot_skeleton.q))
            self.restPoseReward -= np.linalg.norm(self.restPose - self.robot_skeleton.q)*self.restPoseWeight
        
        #total reward        
        reward = reward_ctrl + alive_bonus + reward_progress + reward_prox + self.restPoseReward + self.velocityPenalty + accelerationPenalty
        
        #record rewards for debugging
        self.cumulativeReward += reward
        self.pastReward = reward
        self.distreward1 = reward_dist1
        self.distreward2 = reward_dist2
        self.taureward = reward_ctrl
        self.dispreward1 = reward_progress1
        self.dispreward2 = reward_progress2
        self.proxreward1 = reward_prox1
        self.proxreward2 = reward_prox2
        
        ob = self._get_obs()

        s = self.state_vector()
        
        #update physx capsules
        #self.updateClothCollisionStructures(hapticSensors=True)
        
        #check cloth deformation for termination
        clothDeformation = 0
        if self.simulateCloth is True:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
        
        #check termination conditions
        done = False
        if not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            done = True
            reward -= 500
        elif (clothDeformation > 5):
            done = True
            reward -= 500
        elif self.stableReacher is True:
            a=0
            #this just prevents the target distance from termination
        elif self.targetActive1 and self.targetActive2:
            if -reward_dist1 < 0.1 and -reward_dist2 < 0.1:
                done = True
                reward += 150
        elif self.targetActive1 and -reward_dist1 < 0.1:
            done = True
            reward += 150
        elif self.targetActive2 and -reward_dist2 < 0.1:
            done = True
            reward += 150

        #increment the step counter
        self.numSteps += 1

        #self collision check
        #self.robot_skeleton.set_self_collision_check(True)
        #print("self collision check" + str(self.robot_skeleton.self_collision_check()))
        #print(self.robot_skeleton.contacted_bodies())

        return ob, reward, done, {}

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = self.robot_skeleton.q
        fingertip = np.array([0.0, -0.06, 0.0])

        if self.viewer is not None:
            if self.interactiveTarget is True:
                if self.targetActive1:
                    self.target = self.viewer.interactors[2].frame.org
                if self.targetActive2:
                    self.target2 = self.viewer.interactors[2].frame.org
            self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]

        vec = self.robot_skeleton.bodynodes[8].to_world(fingertip) - self.target
        vec2 = self.robot_skeleton.bodynodes[14].to_world(fingertip) - self.target2
        
        if self.simulateCloth is True:
            f = self.clothScene.getHapticSensorObs()#get force from simulation 
        elif self.sensorNoise is not 0:
            f = np.random.uniform(-self.sensorNoise, self.sensorNoise, f_size)
        else:
            f = np.zeros(f_size)
        
        #print("ID getobs:" + str(self.clothScene.id))
        #print("f: " + str(f))
        #print("len f = " + str(len(f)))
        #return np.concatenate([np.cos(theta), np.sin(theta), self.target, self.robot_skeleton.dq, vec,f]).ravel()
        target1bit = np.array([0.])
        if self.targetActive1:
            target1bit[0] = 1.
        target2bit = np.array([0.])
        if self.targetActive2:
            target2bit[0] = 1.
        #return np.concatenate([np.cos(theta), np.sin(theta), self.robot_skeleton.dq, target1bit, vec, self.target, target2bit, vec2, self.target2, f]).ravel()
        tar = np.array(self.target)
        v = np.array(vec)
        if self.arm == 2:
            tar = np.array(self.target2)
            v = np.array(vec2)

        obs = np.concatenate([np.cos(theta), np.sin(theta), self.robot_skeleton.dq])
        if self.targetActive1 or self.targetActive2:
            obs = np.concatenate([obs, v, tar])
        if self.usePoseTarget is True:
            obs = np.concatenate([obs, self.restPose])
        obs = np.concatenate([obs, f])

        return obs.ravel()
        #return np.concatenate([theta, self.robot_skeleton.dq, vec]).ravel()

    def reset_model(self):
        #print("reset")
        self.cumulativeReward = 0
        self.dart_world.reset()
        self.clothScene.reset()
        self.clothScene.translateCloth(0, np.array([-3.5,0,0]))
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)

        qpos[0] -= 0
        qpos[1] -= 0.
        qpos[2] += 0
        qpos[3] += 0.
        qpos[4] -= 0.
        # qpos[5] += 1
        qpos[5] -= 0.5

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.prevDq = np.array(qvel)

        #set rest pose
        if self.restPoseActive is True:
            self.restPose = qpos
            self.restPose = np.array([0.001827724094, -0.00492083024962, 0.00269002874384, 0.00424451760036, 0.00892373789407, -0.275006149068, 0.421834512474, 0.703907952432, 1.83666666667, -0.00852162044016, 0.00391015137603, 0.00148231514513, -0.00964949311134, 0.00629198543994, 0.0058226973426, 0.00388720644317, 0.0, 0.0073720446693, -0.000369173199674, 0.00479662592095, -0.00559061958513, 0.000216556543277])

            #more edits here if necessary
        
        #reset cloth tube orientation and rotate sphere position
        v1 = np.array([0,0,-1])
        v2 = np.array([-1.,0,0])
        self.random_dir = v2
        if self.simulateCloth is True:   
            if self.rotateCloth is True:
                while True:
                    v2 = self.clothScene.sampleDirections()[0]
                    if np.dot(v2/np.linalg.norm(v2), np.array([0,-1,0.])) < 1:
                        break
            M = self.clothScene.rotateTo(v1,v2)
            self.clothScene.translateCloth(0, np.array([0,0,-0.5]))
            self.clothScene.translateCloth(0, np.array([-0.75,0,0]))
            self.clothScene.translateCloth(0, np.array([0,-0.1,0]))
            self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=random.uniform(0, 6.28), axis=np.array([0,0,1.])))
            self.clothScene.rotateCloth(0, M)
        
        
            #move cloth out of arm range
            self.clothScene.translateCloth(0, np.array([-10.5,0,0]))
        
        #randomize the active target state
        if self.randomActiveTarget:
            r = random.random()
            #print(r)
            if r < 0.25:
                self.targetActive1 = True
                self.targetActive2 = True
            elif r < 0.5:
                self.targetActive1 = False
                self.targetActive2 = False
            elif r < 0.75:
                self.targetActive1 = True
                self.targetActive2 = False
            else:
                self.targetActive1 = False
                self.targetActive2 = True
        
        #old sampling in box
        #'''
        reacher_range = 1.0
        if not self.sampleFromHemisphere:
            if self.targetActive1:
                while True:
                    self.target = self.np_random.uniform(low=-reacher_range, high=reacher_range, size=3)
                    #print('target = ' + str(self.target))
                    if np.linalg.norm(self.target) < reacher_range: break
            if self.targetActive2:
                while True:
                    self.target2= self.np_random.uniform(low=-reacher_range, high=reacher_range, size=3)
                    #print('target = ' + str(self.target))
                    if np.linalg.norm(self.target2) < reacher_range: break
            
        #'''
        
        #sample target from hemisphere
        #if self.sampleFromHemisphere is True:
        #    self.target = self.hemisphereSample(radius=reacher_range, norm=v2)
        #    self.target2 = self.hemisphereSample(radius=reacher_range, norm=v2)
            
        #self.target = np.array([0.,0.,0.])
        
        '''dim = 15
        if(self.reset_number < dim*dim*dim):
            self.target = voxelCenter(dim=dim, radius=0.8, ix=self.reset_number)
            #print(self.target)
        else:
            self.trackSuccess = False'''

        self.dart_world.skeletons[0].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]

        #update physx capsules
        #self.updateClothCollisionStructures(hapticSensors=True)
        #self.clothScene.clearInterpolation()

        #debugging
        self.reset_number += 1
        self.numSteps = 0
        
        obs = self._get_obs()
        
        #self.render()
        if self.simulateCloth is True:
            if np.linalg.norm(obs[-30:]) > 0.00001:
                #print("COLLISION")
                self.reset_model()
        
        if self.trackSuccess is True:
            self.targetHistory.append(self.target)
            self.successHistory.append(False)

        return self._get_obs()

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        a=0
        #collision spheres creation
        fingertip = np.array([0.0, -0.06, 0.0])
        z = np.array([0.,0,0])
        cs0 = self.robot_skeleton.bodynodes[1].to_world(z)
        cs1 = self.robot_skeleton.bodynodes[2].to_world(z)
        cs2 = self.robot_skeleton.bodynodes[16].to_world(z)
        cs3 = self.robot_skeleton.bodynodes[16].to_world(np.array([0,0.175,0]))
        cs4 = self.robot_skeleton.bodynodes[4].to_world(z)
        cs5 = self.robot_skeleton.bodynodes[6].to_world(z)
        cs6 = self.robot_skeleton.bodynodes[7].to_world(z)
        cs7 = self.robot_skeleton.bodynodes[8].to_world(z)
        cs8 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        cs9 = self.robot_skeleton.bodynodes[10].to_world(z)
        cs10 = self.robot_skeleton.bodynodes[12].to_world(z)
        cs11 = self.robot_skeleton.bodynodes[13].to_world(z)
        cs12 = self.robot_skeleton.bodynodes[14].to_world(z)
        cs13 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        csVars0 = np.array([0.15, -1, -1, 0,0,0])
        csVars1 = np.array([0.07, -1, -1, 0,0,0])
        csVars2 = np.array([0.1, -1, -1, 0,0,0])
        csVars3 = np.array([0.1, -1, -1, 0,0,0])
        csVars4 = np.array([0.065, -1, -1, 0,0,0])
        csVars5 = np.array([0.05, -1, -1, 0,0,0])
        csVars6 = np.array([0.0365, -1, -1, 0,0,0])
        csVars7 = np.array([0.04, -1, -1, 0,0,0])
        csVars8 = np.array([0.036, -1, -1, 0,0,0])
        csVars9 = np.array([0.065, -1, -1, 0,0,0])
        csVars10 = np.array([0.05, -1, -1, 0,0,0])
        csVars11 = np.array([0.0365, -1, -1, 0,0,0])
        csVars12 = np.array([0.04, -1, -1, 0,0,0])
        csVars13 = np.array([0.036, -1, -1, 0,0,0])
        collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3, cs4, csVars4, cs5, csVars5, cs6, csVars6, cs7, csVars7, cs8, csVars8, cs9, csVars9, cs10, csVars10, cs11, csVars11, cs12, csVars12, cs13, csVars13]).ravel()
        #collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        
        self.clothScene.setCollisionSpheresInfo(collisionSpheresInfo)
        
        if capsules is True:
            #collision capsules creation
            collisionCapsuleInfo = np.zeros((14,14))
            collisionCapsuleInfo[0,1] = 1
            collisionCapsuleInfo[1,2] = 1
            collisionCapsuleInfo[1,4] = 1
            collisionCapsuleInfo[1,9] = 1
            collisionCapsuleInfo[2,3] = 1
            collisionCapsuleInfo[4,5] = 1
            collisionCapsuleInfo[5,6] = 1
            collisionCapsuleInfo[6,7] = 1
            collisionCapsuleInfo[7,8] = 1
            collisionCapsuleInfo[9,10] = 1
            collisionCapsuleInfo[10,11] = 1
            collisionCapsuleInfo[11,12] = 1
            collisionCapsuleInfo[12,13] = 1
            self.clothScene.setCollisionCapsuleInfo(collisionCapsuleInfo)
            
        if hapticSensors is True:
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.33), LERP(cs0, cs1, 0.66), cs1, LERP(cs1, cs2, 0.33), LERP(cs1, cs2, 0.66), cs2, LERP(cs2, cs3, 0.33), LERP(cs2, cs3, 0.66), cs3])
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, LERP(cs2, cs3, 0.25), LERP(cs2, cs3, 0.5), LERP(cs2, cs3, 0.75), cs3])
            hapticSensorLocations = np.concatenate([cs0, cs1, cs2, cs3, cs4, LERP(cs4, cs5, 0.33), LERP(cs4, cs5, 0.66), cs5, LERP(cs5, cs6, 0.33), LERP(cs5,cs6,0.66), cs6, cs7, cs8, cs9, LERP(cs9, cs10, 0.33), LERP(cs9, cs10, 0.66), cs10, LERP(cs10, cs11, 0.33), LERP(cs10, cs11, 0.66), cs11, cs12, cs13])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)
            
    def getViewer(self, sim, title=None, extraRenderFunc=None, inputFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction, self.inputFunc)
        
    def hemisphereSample(self, radius=1, norm=np.array([0,0,1.]), frustrum = 0.6):
        p = norm
        while True:
            p = self.np_random.uniform(low=-radius, high=radius, size=3)
            p_n = np.linalg.norm(p)
            if p_n < radius:
                if(np.dot(p/p_n, norm) > frustrum):
                    return p

        
    def extraRenderFunction(self):
        #print("extra render function")
        
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()
        
        #render target 2
        GL.glColor3d(0.8,0.,0)
        GL.glPushMatrix()
        GL.glTranslated(self.target2[0], self.target2[1], self.target2[2])
        GLUT.glutSolidCube(0.1, 10,10)
        GL.glPopMatrix()
        
        #if targets are active, render guidelines
        fingertip = np.array([0.0, -0.06, 0.0])
        if self.targetActive1:
            wef1 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
            GL.glColor3d(0.0,0.0,0.8)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3d(self.target[0], self.target[1], self.target[2])
            GL.glVertex3d(wef1[0],wef1[1],wef1[2])
            GL.glEnd()
            
        if self.targetActive2:
            wef2 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
            GL.glColor3d(0.8,0.0,0.0)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3d(self.target2[0], self.target2[1], self.target2[2])
            GL.glVertex3d(wef2[0],wef2[1],wef2[2])
            GL.glEnd()
        
        if self.renderSuccess is True or self.renderFailure is True:
            for i in range(len(self.targetHistory)):
                p = self.targetHistory[i]
                s = self.successHistory[i]
                if (s and self.renderSuccess) or (not s and self.renderFailure):
                    #print("draw")
                    GL.glColor3d(1,0.,0)
                    if s is True:
                        GL.glColor3d(0,1.,0)
                    GL.glPushMatrix()
                    GL.glTranslated(p[0], p[1], p[2])
                    GLUT.glutSolidSphere(self.successSampleRenderSize, 10,10)
                    GL.glPopMatrix()
        
        #draw hemisphere samples for target sampling
        '''
        GL.glColor3d(0,1,0)
        for i in range(1000):
            p = self.hemisphereSample(radius=1.4, norm=self.random_dir)
            #p=np.array([0,0,0.])
            #while True:
            #    p = self.np_random.uniform(low=-1.5, high=1.5, size=3)
            #    if np.linalg.norm(p) < 1.5: break
            GL.glPushMatrix()
            GL.glTranslated(p[0], p[1], p[2])
            GLUT.glutSolidSphere(0.01, 10,10)
            GL.glPopMatrix()
        '''
        
        #render target vector
        '''
        ef = self.robot_skeleton.bodynodes[5].to_world(np.array([0,-0.06,0]))
        #vec = ef - self.target
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(ef[0], ef[1], ef[2])
        GL.glVertex3d(self.target[0], self.target[1], self.target[2])
        GL.glEnd()
        '''
        #print("ID:" + str(self.clothScene.id))
        m_viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        
        if self.renderRewards:
            self.clothScene.drawText(x=15, y=m_viewport[3]-15*1, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0.,0,0))
            self.clothScene.drawText(x=15, y=m_viewport[3]-15*2, text="Past Reward = " + str(self.pastReward), color=(0.,0,0))
            self.clothScene.drawText(x=15, y=m_viewport[3]-15*3, text="Dist Reward 1 = " + str(self.distreward1), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*4, text="Dist Reward 2 = " + str(self.distreward2), color=(0.,0,0))
            self.clothScene.drawText(x=15, y=m_viewport[3]-15*5, text="Tau Reward = " + str(self.taureward), color=(0.,0,0))
            self.clothScene.drawText(x=15, y=m_viewport[3]-15*6, text="Disp Reward 1 = " + str(self.dispreward1), color=(0.,0,0))
            self.clothScene.drawText(x=15, y=m_viewport[3]-15*7, text="Vel Penalty = " + str(self.velocityPenalty), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*7, text="Disp Reward 2 = " + str(self.dispreward2), color=(0.,0,0))
            self.clothScene.drawText(x=15, y=m_viewport[3]-15*8, text="Prox Reward 1 = " + str(self.proxreward1), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*9, text="Prox Reward 2 = " + str(self.proxreward2), color=(0.,0,0))
            self.clothScene.drawText(x=15, y=m_viewport[3]-15*10, text="Rest-pose Reward = " + str(self.restPoseReward), color=(0., 0, 0))
        
        textX = 15.
        if self.renderForceText:
            HSF = self.clothScene.getHapticSensorObs()
            #print("HSF: " + str(HSF))
            for i in range(self.clothScene.getNumHapticSensors()):
                #print("i = " + str(i))
                #print("HSL[i] = " + str(HSL[i*3:i*3+3]))
                #print("HSF[i] = " + str(HSF[i*3:i*3+3]))
                self.clothScene.drawText(x=textX, y=60.+15*i, text="||f[" + str(i) + "]|| = " + str(np.linalg.norm(HSF[3*i:3*i+3])), color=(0.,0,0))
            textX += 160
        
        topLeft = np.array([2.,self.viewer.viewport[3]-10-150])
        self.viewer.interactors[4].topLeft = np.array(topLeft)
        self.viewer.interactors[4].boxesDefined = False
        renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False, _topLeft=topLeft)

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(self, repeat)

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            self._get_viewer().scene.tb.trans[2] = -3.5
            self._get_viewer().scene.tb._set_theta(180)
            self._get_viewer().scene.tb._set_phi(180)
        self.track_skeleton_id = 0

    '''def skelVoxelAnalysis(self, dim, radius, samplerate=0.1, depth=0, efn=5, efo=np.array([0.,0,0]), displayReachable = True, displayUnreachable=True):
        #initialize binary voxel structure (0)
        if depth == 0:
            self.voxels = np.zeros(dim*dim*dim)
            
        #step through all joint samples
        qpos = self.robot_skeleton.q
        low  = self.robot_skeleton.dof(depth).position_lower_limit()
        high = self.robot_skeleton.dof(depth).position_upper_limit()
        qpos[depth] = low
        self.robot_skeleton.set_positions(qpos)
        samples = int((high-low)/samplerate)
        if(samples == 0):
            samples = 1
        #print(samples)
        for i in range(samples):
            if depth < 5:
                print("[" + str(depth) + "]" + str(i) + "/" + str(samples))
            qpos = self.robot_skeleton.q
            qpos[depth] = low + i*samplerate
            if depth < len(qpos)-1:
                self.skelVoxelAnalysis(dim, radius, samplerate, depth+1)
            else:
                r = np.array([radius, radius, radius])
                ef = self.robot_skeleton.bodynodes[efn].to_world(efo)
                efp = (ef + r)/(2.*r) #transform into voxel grid space (centered at [-r,-r,-r])
                ix = math.floor(efp[0]*dim) + math.floor(efp[1]*dim*dim) + math.floor(efp[2]*dim*dim*dim)
                self.voxels[ix] += 1
        
        
        #TODO: add voxels to visualization
        if depth == 0:
            for v in range(len(self.voxels)):
                if self.voxels[v] > 0 and displayReachable:
                    self.targetHistory.append(self.voxelCenter(dim, radius, v))
                    self.successHistory.append(True)
                elif displayUnreachable:
                    self.targetHistory.append(self.voxelCenter(dim, radius, v))
                    self.successHistory.append(False)
                
    '''    
        
        
def LERP(p0, p1, t):
    return p0 + (p1-p0)*t

    
def ROM1(low, high, t):
    return LERP(low, high, t)
    
def ROM2(l1, h1, l2, h2, t):
    #print("l1 " + str(l1) + " h1 " + str(h1) + " l2 " + str(l2) + " h2 " +  str(h2) + " t "  + str(t))
    #print("LERP(l1, h1, t) = " + str(LERP(l1, h1, t)))
    val = np.array([0.,0.])
    if t < 0.25:
        val[0] = LERP(l1, h1, t*4.0)
        val[1] = l2
    elif t<0.5:
        val[0] = h1
        val[1] = LERP(l2,h2,(t-0.25)*4.0)
    elif t<0.75:
        val[0] = LERP(h1, l1, (t-0.5)*4.0)
        val[1] = h2
    else:
        val[0] = l1
        val[1] = LERP(h2,l2,(t-0.75)*4.0)
    return val
    
