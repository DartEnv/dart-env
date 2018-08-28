# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.upperbodydatadriven_cloth_base import *
import random
import time
import math

import pydart2.joint as Joint

import pybullet as p
import pybullet_data
import os

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils
from pyPhysX.pyutils import LERP
import pyPhysX.renderUtils
import pyPhysX.meshgraph as meshgraph
from pyPhysX.clothfeature import *

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

class Controller(object):
    def __init__(self, env, skel, policyfilename=None, name=None, obs_subset=[]):
        self.env = env #needed to set env state variables on setup for use
        self.skel = skel
        self.name = name
        prefix = os.path.dirname(os.path.abspath(__file__))
        prefix = os.path.join(prefix, '../../../../rllab/data/local/experiment/')
        if name is None:
            self.name = policyfilename
        self.policy = None
        if policyfilename is not None:
            self.policy = pickle.load(open(prefix+policyfilename + "/policy.pkl", "rb"))
        self.obs_subset = obs_subset #list of index,length tuples to slice obs for input

    def query(self, obs):
        obs_subset = np.array([])
        for s in self.obs_subset:
            obs_subset = np.concatenate([obs_subset, obs[s[0]:s[0]+s[1]]]).ravel()
        a, a_info = self.policy.get_action(obs_subset)
        a = a_info['mean']
        return a

    def setup(self):
        print("base setup ... overwrite this for specific control requirements")
        #TODO: subclasses for setup requirements

    def update(self):
        print("default update")
        #TODO: subclasses update targets, etc...

    def transition(self):
        #return true when a controller detects task completion to transition to the next controller
        return False

class SPDController(Controller):
    def __init__(self, env, skel, target=None, timestep=0.01):
        obs_subset = []
        policyfilename = None
        name = "SPD"
        self.target = target
        Controller.__init__(self, env, skel, policyfilename, name, obs_subset)

        self.h = timestep
        #self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs-6
        self.qhat = self.skel.q
        self.Kp = np.diagflat([30000.0] * (ndofs))
        self.Kd = np.diagflat([50.0] * (ndofs))

        #self.Kd[0, 6] = 1.0

        self.Kd[6,6] = 1.0

        #self.Kp[0][0] = 2000.0
        #self.Kd[0][0] = 100.0
        #self.Kp[1][1] = 2000.0
        #self.Kp[2][2] = 2000.0
        #self.Kd[2][2] = 100.0
        #self.Kp[3][3] = 2000.0
        #self.Kp[4][4] = 2000.0

        '''
        for i in range(ndofs):
            if i ==9 or i==10 or i==17 or i==18:
                self.Kd[i][i] *= 0.01
                self.Kp[i][i] *= 0.01
        '''

        #print(self.Kp)
        self.preoffset = 0.0

    def setup(self):
        #reset the target
        #cur_q = np.array(self.skel.q)
        #self.env.loadCharacterState(filename="characterState_regrip")
        self.target = np.array(self.skel.q[6:])
        #self.env.restPose = np.array(self.target)
        #self.target = np.array(self.skel.q)
        #self.env.robot_skeleton.set_positions(cur_q)

        a=0

    def update(self):
        #if self.env.handleNode is not None:
        #    self.env.handleNode.clearHandles();
        #    self.env.handleNode = None
        a=0

    def transition(self):
        return False

    def query(self, obs):
        #test adaptive gains
        ndofs = self.skel.ndofs - 6
        self.Kd = np.diagflat([300.0] * (ndofs))
        dif = self.skel.q[6:]-self.target
        for i in range(7):
            dm = abs(dif[i])
            if(dm > 0.75):
                self.Kd[i,i] = 1.0
            elif(dm > 0.2):
                self.Kd[i, i] = LERP(300.0, 1.0, (dm-0.2)/(0.55))
            #print("dm: " + str(dm) + " kd = " + str(self.Kd[i,i]))

        self.Kd[6,6] = 1.0

        #SPD
        self.qhat = self.target
        skel = self.skel
        p = -self.Kp.dot(skel.q[6:] + skel.dq[6:] * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq[6:])
        b = -skel.c[6:] + p + d + skel.constraint_forces()[6:]
        A = skel.M[6:, 6:] + self.Kd * self.h

        #print(np.linalg.cond(A))
        #TODO: near singular matrix check ... remove for speed
        if not np.linalg.cond(A) < 1/sys.float_info.epsilon:
            print("Near singular...")

        x = np.linalg.solve(A, b)

        #invM = np.linalg.inv(A)
        #x = invM.dot(b)
        #tau = p - self.Kd.dot(skel.dq[6:] + x * self.h)
        tau = p + d - self.Kd.dot(x) * self.h
        return tau

class DartClothUpperBodyDataDrivenRigidClothSawyerEnv(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = False
        self.renderCloth = False
        dt = 0.002
        frameskip = 20

        #observation terms
        self.featureInObs   = True  # if true, feature centroid location and displacement from ef are observed
        self.oracleInObs    = True  # if true, oracle vector is in obs
        self.contactIDInObs = True  # if true, contact ids are in obs
        self.hapticsInObs   = True  # if true, haptics are in observation
        self.prevTauObs     = False  # if true, previous action in observation

        #reward flags
        self.uprightReward              = True  #if true, rewarded for 0 torso angle from vertical
        self.stableHeadReward           = True  # if True, rewarded for - head/torso angle
        self.elbowFlairReward           = False
        self.limbProgressReward         = True  # if true, the (-inf, 1] plimb progress metric is included in reward
        self.oracleDisplacementReward   = True  # if true, reward ef displacement in the oracle vector direction
        self.contactGeoReward           = True  # if true, [0,1] reward for ef contact geo (0 if no contact, 1 if limbProgress > 0).
        self.deformationPenalty         = True
        self.restPoseReward             = True

        self.uprightRewardWeight              = 5  #if true, rewarded for 0 torso angle from vertical
        self.stableHeadRewardWeight           = 1
        self.elbowFlairRewardWeight           = 1
        self.limbProgressRewardWeight         = 10  # if true, the (-inf, 1] plimb progress metric is included in reward
        self.oracleDisplacementRewardWeight   = 50  # if true, reward ef displacement in the oracle vector direction
        self.contactGeoRewardWeight           = 2  # if true, [0,1] reward for ef contact geo (0 if no contact, 1 if limbProgress > 0).
        self.deformationPenaltyWeight         = 5
        self.restPoseRewardWeight             = 1

        #other flags
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.collarTermination  = False  #if true, rollout terminates when collar is off the head/neck
        self.sleeveEndTerm      = False  #if true, terminate the rollout if the arm enters the end of sleeve feature before the beginning (backwards dressing)
        self.elbowFirstTerm     = True #if true, terminate when any limb enters the feature before the hand

        #other variables
        self.prevTau = None
        self.elbowFlairNode = 10
        self.maxDeformation = 30.0
        self.restPose = None
        self.prevOracle = np.zeros(3)
        self.prevAvgGeodesic = None
        self.localLeftEfShoulder1 = None
        self.limbProgress = 0
        self.previousDeformationReward = 0
        self.handFirst = False #once the hand enters the feature, switches to true
        self.state_save_directory = "saved_control_states/"
        self.fingertip = np.array([0,-0.085,0])
        self.ef_accuracy_info = {'best':0, 'worst':0, 'total':0, 'average':0 }

        #linear track variables
        self.trackInitialRange = [np.array([0.42, 0.2,-0.7]), np.array([-0.21, -0.3, -0.8])]
        self.trackEndRange = [np.array([0.21, 0.1,-0.1]),np.array([0.21, 0.1,-0.1])]
        self.trackTraversalSteps = 250 #seconds for track traversal
        self.linearTrackActive = False
        self.linearTrackTarget = np.zeros(3)
        self.linearTrackOrigin = np.zeros(3)

        # limb progress tracking
        self.limbProgressGraphing = False
        self.limbProgressGraph = None
        if(self.limbProgressGraphing):
            self.limbProgressGraph = pyutils.LineGrapher(title="Limb Progress")

        # restPose error tracking
        self.restPoseErrorGraphing = False
        self.restPoseErrorGraph = None
        if (self.restPoseErrorGraphing):
            self.restPoseErrorGraph = pyutils.LineGrapher(title="Rest Pose Error")

        self.handleNode = None
        self.updateHandleNodeFrom = 12  # left fingers

        self.actuatedDofs = np.arange(22)
        observation_size = len(self.actuatedDofs)*3 #q(sin,cos), dq
        if self.prevTauObs:
            observation_size += len(self.actuatedDofs)
        if self.hapticsInObs:
            observation_size += 66
        if self.featureInObs:
            observation_size += 6
        if self.oracleInObs:
            observation_size += 3
        if self.contactIDInObs:
            observation_size += 22

        # initialize the Sawyer variables
        self.SPDController = None
        self.sawyer_skel = None
        self.ikPath = pyutils.Spline()
        self.ikPathTimeScale = 0.0025  # relationship between number of steps and spline time
        self.ikTarget = np.array([0.5, 0, 0])
        self.trackPosePath = False #if true, no IK, track a pose path
        self.kinematicIK = False
        self.root_adjustment = False
        self.ikOrientation = False
        self.maxSawyerReach = 1.0 #omni-directional reach (from 2nd dof)
        self.previousIKResult = np.zeros(7)
        self.sawyer_root_dofs = np.array([-1.2, -1.2, -1.2, 0, -0.1, -0.9]) #values for the fixed 6 dof root transformation
        self.sawyer_rest = np.array([0, 0, 0, 0, 0, 0, 0])
        self.rigidClothFrame = pyutils.BoxFrame(c0=np.array([0.1,0.2,0.001]),c1=np.array([-0.1,0,-0.001]))
        self.rigidClothTargetFrame = pyutils.BoxFrame(c0=np.array([0.1,0.2,0.001]),c1=np.array([-0.1,0,-0.001]))
        self.renderIKGhost = True
        self.renderSawyerReach = False
        self.posePath = pyutils.Spline()


        # SPD error graphing per dof
        self.graphSPDError = False
        self.SPDErrorGraph = None
        if self.graphSPDError:
            self.SPDErrorGraph = pyutils.LineGrapher(title="SPD Error Violation", numPlots=7, legend=True)
            for i in range(len(self.SPDErrorGraph.labels)):
                self.SPDErrorGraph.labels[i] = str(i)

        #setup pybullet
        print("Setting up pybullet")
        self.pyBulletPhysicsClient = p.connect(p.DIRECT)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.pyBulletSawyer = p.loadURDF(dir_path + '/assets/sawyer_description/urdf/sawyer_arm.urdf')
        print("Sawyer bodyID: " + str(self.pyBulletSawyer))
        print("Number of pybullet joints: " + str(p.getNumJoints(self.pyBulletSawyer)))
        for i in range(p.getNumJoints(self.pyBulletSawyer)):
            jinfo = p.getJointInfo(self.pyBulletSawyer, i)
            print(" " + str(jinfo[0]) + " " + str(jinfo[1]) + " " + str(jinfo[2]) + " " + str(jinfo[3]) + " " + str(
                jinfo[12]))

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1280,720),
                                                          clothMeshFile="fullgown1.obj",
                                                          clothMeshStateFile = "hanginggown.obj",
                                                          #clothMeshStateFile = "objFile_1starmin.obj",
                                                          clothScale=np.array([1.3, 1.3, 1.3]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation,
                                                          dt=dt,
                                                          frameskip=frameskip)

        #initialize the Sawyer robot
        sawyerFilename = os.path.join(os.path.dirname(__file__), "assets", 'sawyer_description/urdf/sawyer_arm_hoop.urdf')
        self.dart_world.add_skeleton(filename=sawyerFilename)
        for s in self.dart_world.skeletons:
            print(s)
        self.sawyer_skel = self.dart_world.skeletons[2]
        print("Sawyer Robot info:")
        print(" BodyNodes: ")
        for bodynode in self.sawyer_skel.bodynodes:
            print("     : " + bodynode.name)

        print(" Joints: ")
        for joint in self.sawyer_skel.joints:
            print("     : " + joint.name)
            joint.set_position_limit_enforced()

        print(" Dofs: ")
        for dof in self.sawyer_skel.dofs:
            print("     : " + dof.name)
            print("         llim: " + str(dof.position_lower_limit()) + ", ulim: " + str(dof.position_upper_limit()))
            # print("         damping: " + str(dof.damping_coefficient()))
            dof.set_damping_coefficient(2.0)
        self.sawyer_skel.dofs[-1].set_damping_coefficient(0.5)
        self.sawyer_skel.dofs[-2].set_damping_coefficient(0.5)
        self.sawyer_skel.dofs[-3].set_damping_coefficient(1.0)

        #self.sawyer_skel.dofs[-1].set_damping_coefficient(0.1)
        #self.sawyer_skel.dofs[-2].set_damping_coefficient(0.1)
        self.sawyer_skel.joints[0].set_actuator_type(Joint.Joint.LOCKED)

        #compute the joint ranges for null space IK
        self.sawyer_dof_llim = np.zeros(7)
        self.sawyer_dof_ulim = np.zeros(7)
        self.sawyer_dof_jr = np.zeros(7)
        for i in range(7):
            self.sawyer_dof_llim[i] = self.sawyer_skel.dofs[i+6].position_lower_limit()
            self.sawyer_dof_ulim[i] = self.sawyer_skel.dofs[i+6].position_upper_limit()
            self.sawyer_dof_jr[i] = self.sawyer_dof_ulim[i] - self.sawyer_dof_llim[i]
            #self.sawyer_dof_jr[i] = 6.28
        #print("Sawyer mobile? " + str(self.sawyer_skel.is_mobile()))

        # initialize the controller
        self.SPDController = SPDController(self, self.sawyer_skel, timestep=frameskip*dt)

        #clothing features
        #self.sleeveRVerts = [46, 697, 1196, 696, 830, 812, 811, 717, 716, 718, 968, 785, 1243, 783, 1308, 883, 990, 739, 740, 742, 1318, 902, 903, 919, 737, 1218, 736, 1217]
        self.sleeveLVerts = [413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531]
        self.sleeveLMidVerts = [413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531]
        self.sleeveLEndVerts = [413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531]
        #self.sleeveRMidVerts = [1054, 1055, 1057, 1058, 1060, 1061, 1063, 1052, 1051, 1049, 1048, 1046, 1045, 1043, 1042, 1040, 1039, 734, 732, 733]
        #self.sleeveREndVerts = [228, 1059, 229, 1062, 230, 1064, 227, 1053, 226, 1050, 225, 1047, 224, 1044, 223, 1041, 142, 735, 141, 1056]
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveLEndFeature = ClothFeature(verts=self.sleeveLEndVerts, clothScene=self.clothScene)
        self.sleeveLMidFeature = ClothFeature(verts=self.sleeveLMidVerts, clothScene=self.clothScene)

        self.simulateCloth = clothSimulation
        if self.simulateCloth:
            self.handleNode = HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975]))

        if not self.renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        for i in range(len(self.robot_skeleton.dofs)):
            self.robot_skeleton.dofs[i].set_damping_coefficient(3.0)

        # load rewards into the RewardsData structure
        if self.uprightReward:
            self.rewardsData.addReward(label="upright", rmin=-2.5, rmax=0, rval=0, rweight=self.uprightRewardWeight)

        if self.stableHeadReward:
            self.rewardsData.addReward(label="stable head",rmin=-1.2,rmax=0,rval=0, rweight=self.stableHeadRewardWeight)

        if self.elbowFlairReward:
            self.rewardsData.addReward(label="elbow flair", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.elbowFlairRewardWeight)

        if self.limbProgressReward:
            self.rewardsData.addReward(label="limb progress", rmin=-1.0, rmax=1.0, rval=0,
                                       rweight=self.limbProgressRewardWeight)

        if self.oracleDisplacementReward:
            self.rewardsData.addReward(label="oracle", rmin=-0.1, rmax=0.1, rval=0,
                                       rweight=self.oracleDisplacementRewardWeight)

        if self.contactGeoReward:
            self.rewardsData.addReward(label="contact geo", rmin=0, rmax=1.0, rval=0,
                                       rweight=self.contactGeoRewardWeight)

        if self.deformationPenalty:
            self.rewardsData.addReward(label="deformation", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.deformationPenaltyWeight)

        if self.restPoseReward:
            self.rewardsData.addReward(label="rest pose", rmin=-51.0, rmax=0, rval=0,
                                       rweight=self.restPoseRewardWeight)

        #self.loadCharacterState(filename="characterState_1starmin")

    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        #update features
        if self.sleeveLSeamFeature is not None:
            self.sleeveLSeamFeature.fitPlane()
        if self.sleeveLEndFeature is not None:
            self.sleeveLEndFeature.fitPlane()
        if self.sleeveLMidFeature is not None:
            self.sleeveLMidFeature.fitPlane()

        #update handle nodes
        if self.handleNode is not None:
            #if self.updateHandleNodeFrom >= 0:
            #    self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            #TODO: linear track
            if self.linearTrackActive:
                self.handleNode.org = LERP(self.linearTrackOrigin, self.linearTrackTarget, self.numSteps/self.trackTraversalSteps)
            self.handleNode.step()

        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        self.localLeftEfShoulder1 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip1)  # right fingertip in right shoulder local frame
        a=0

        if(self.trackPosePath):
            self.previousIKResult = self.posePath.pos(self.numSteps * self.ikPathTimeScale)
        else:
            #sawyer IK
            self.ikTarget = self.ikPath.pos(self.numSteps * self.ikPathTimeScale)

            self.rigidClothTargetFrame.setFromDirectionandUp(dir=-self.ikTarget, up=np.array([0, -1.0, 0]),
                                                             org=self.ikTarget)
            tar_quat = self.rigidClothTargetFrame.quat
            tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
            tar_dir = -self.ikTarget/np.linalg.norm(self.ikTarget)
            #standard IK
            #result = p.calculateInverseKinematics(self.pyBulletSawyer, 12, self.ikTarget-self.sawyer_root_dofs[3:])

            #IK with joint limits
            #print("computing IK")
            result = None
            if(self.ikOrientation):
                result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletSawyer,
                                                      endEffectorLinkIndex=12,
                                                      targetPosition=self.ikTarget-self.sawyer_root_dofs[3:],
                                                      targetOrientation=tar_quat,
                                                      #targetOrientation=tar_dir,
                                                      lowerLimits=self.sawyer_dof_llim.tolist(),
                                                      upperLimits=self.sawyer_dof_ulim.tolist(),
                                                      jointRanges=self.sawyer_dof_jr.tolist(),
                                                      restPoses=self.sawyer_skel.q[6:].tolist()
                                                      )
            else:
                result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletSawyer,
                                                      endEffectorLinkIndex=12,
                                                      targetPosition=self.ikTarget-self.sawyer_root_dofs[3:],
                                                      #targetOrientation=tar_quat,
                                                      #targetOrientation=tar_dir,
                                                      lowerLimits=self.sawyer_dof_llim.tolist(),
                                                      upperLimits=self.sawyer_dof_ulim.tolist(),
                                                      jointRanges=self.sawyer_dof_jr.tolist(),
                                                      restPoses=self.sawyer_skel.q[6:].tolist()
                                                      )
            #print("computed IK result: " + str(result))
            self.previousIKResult = np.array(result)
            self.setPosePyBullet(result)
        #self.sawyer_skel.set_positions(np.concatenate([np.array([0, 0, 0, 0, 0.25, -0.9]), result]))
        if(self.root_adjustment):
            self.sawyer_skel.set_positions(np.concatenate([np.array(self.sawyer_root_dofs), np.zeros(7)]))
        elif (self.kinematicIK):
            # kinematic
            self.sawyer_skel.set_positions(np.concatenate([np.array(self.sawyer_root_dofs), result]))
        else:
            # SPD (dynamic)
            if self.SPDController is not None:
                self.SPDController.target = self.previousIKResult
                tau = np.concatenate([np.zeros(6), self.SPDController.query(obs=None)])
                #self.do_simulation(tau, self.frame_skip)
                self.sawyer_skel.set_forces(tau)

            #check the Sawyer arm for joint, velocity and torque limits
            tau = self.sawyer_skel.forces()
            tau_upper_lim = self.sawyer_skel.force_upper_limits()
            tau_lower_lim = self.sawyer_skel.force_lower_limits()
            vel = self.sawyer_skel.velocities()
            pos = self.sawyer_skel.positions()
            pos_upper_lim = self.sawyer_skel.position_upper_limits()
            pos_lower_lim = self.sawyer_skel.position_lower_limits()
            for i in range(len(tau)):
                if(tau[i] > tau_upper_lim[i]):
                    #print(" tau["+str(i)+"] close to upper lim: " + str(tau[i]) + "|"+ str(tau_upper_lim[i]))
                    tau[i] = tau_upper_lim[i]
                if (tau[i] < tau_lower_lim[i]):
                    #print(" tau[" + str(i) + "] close to lower lim: " + str(tau[i]) + "|" + str(tau_lower_lim[i]))
                    tau[i] = tau_lower_lim[i]
                #if(pos_upper_lim[i]-pos[i] < 0.1):
                #    print(" pos["+str(i)+"] close to upper lim: " + str(pos[i]) + "|"+ str(pos_upper_lim[i]))
                #if (pos[i] - pos_lower_lim[i] < 0.1):
                #    print(" pos[" + str(i) + "] close to lower lim: " + str(pos[i]) + "|" + str(pos_lower_lim[i]))

            for i in range(7):
                if(self.previousIKResult[i] > pos_upper_lim[i+6]):
                    print("invalid IK solution: result["+str(i)+"] over upper limit: " + str(self.previousIKResult[i]) + "|"+ str(pos_upper_lim[i+6]))
                if(self.previousIKResult[i] < pos_lower_lim[i+6]):
                    print("invalid IK solution: result["+str(i)+"] under lower limit: " + str(self.previousIKResult[i]) + "|"+ str(pos_lower_lim[i+6]))

            self.sawyer_skel.set_forces(tau)
                        #print(self.robot_skeleton.q)
        #self.maxSawyerReach = max(self.maxSawyerReach, np.linalg.norm(self.sawyer_skel.bodynodes[3].to_world(np.zeros(3)) - self.sawyer_skel.bodynodes[13].to_world(np.zeros(3))))
        #print("maxSawyerReach = " + str(self.maxSawyerReach))

    def checkTermination(self, tau, s, obs):

        #check joint velocity within limits
        for vx in range(len(self.sawyer_skel.dq)):
            #print("vx: " + str(self.sawyer_skel.dq[vx]) + " | " + str(self.sawyer_skel.dofs[vx].velocity_upper_limit()))
            if(abs(self.sawyer_skel.dq[vx]) > self.sawyer_skel.dofs[vx].velocity_upper_limit()):
                print("Invalid velocity: " + str(vx) + ": " + str(self.sawyer_skel.dq[vx]) + " | " + str(self.sawyer_skel.dofs[vx].velocity_upper_limit()))
        #compute ef_accuracy here (after simulation step)
        #self.ef_accuracy_info = {'best': 0, 'worst': 0, 'total': 0, 'average': 0}
        if(not self.trackPosePath):
            ef_accuracy = np.linalg.norm(self.sawyer_skel.bodynodes[13].to_world(np.zeros(3)) - self.ikTarget)
            if(self.numSteps == 0):
                self.ef_accuracy_info['best'] = ef_accuracy
                self.ef_accuracy_info['worst'] = ef_accuracy
                self.ef_accuracy_info['total'] = ef_accuracy
                self.ef_accuracy_info['average'] = ef_accuracy
            else:
                self.ef_accuracy_info['best'] = min(ef_accuracy, self.ef_accuracy_info['best'])
                self.ef_accuracy_info['worst'] = max(ef_accuracy, self.ef_accuracy_info['worst'])
                self.ef_accuracy_info['total'] += ef_accuracy
                self.ef_accuracy_info['average'] = self.ef_accuracy_info['total']/self.numSteps

        pose_error = self.sawyer_skel.q[6:] - self.previousIKResult
        if self.graphSPDError:
            self.SPDErrorGraph.addToLinePlot(data=pose_error.tolist())

        #set shape frame for rigid cloth
        hn = self.sawyer_skel.bodynodes[14] # hoop 1 node
        self.rigidClothFrame.setTransform(hn.world_transform())
        #self.rigidClothTargetFrame.setFromDirectionandUp(dir=-self.rigidClothFrame.org, up=np.array([0, -1.0, 0]), org=self.rigidClothFrame.org)
        #R = pyutils.rotateX(1.56)
        #self.rigidClothTargetFrame.applyRotationMatrix(R)

        # save state for rendering
        if self.recordForRendering:
            fname = self.recordForRenderingOutputPrefix
            gripperfname_ix = fname + "_grip%05d" % self.renderSaveSteps
            self.saveGripperState(gripperfname_ix)

        #check the termination conditions and return: done,reward
        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))
        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, -500
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -500
        elif self.sleeveEndTerm and self.limbProgress <= 0 and self.simulateCloth:
            limbInsertionError = pyutils.limbFeatureProgress(
                limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL,
                                                  offset=np.array([0, -0.095, 0])), feature=self.sleeveLEndFeature)
            if limbInsertionError > 0:
                return True, -500
        elif self.elbowFirstTerm and self.simulateCloth and not self.handFirst:
            if self.limbProgress > 0 and self.limbProgress < 0.14:
                self.handFirst = True
            else:
                limbInsertionError = pyutils.limbFeatureProgress(
                    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL[:3]),
                    feature=self.sleeveLSeamFeature)
                if limbInsertionError > 0:
                    return True, -500
        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip2 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        localLeftEfShoulder2 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip2)  # right fingertip in right shoulder local frame

        self.prevTau = tau
        reward_record = []

        # reward for maintaining posture
        reward_upright = 0
        if self.uprightReward:
            reward_upright = max(-2.5, -abs(self.robot_skeleton.q[0]) - abs(self.robot_skeleton.q[1]))
            reward_record.append(reward_upright)

        reward_stableHead = 0
        if self.stableHeadReward:
            reward_stableHead = max(-1.2, -abs(self.robot_skeleton.q[19]) - abs(self.robot_skeleton.q[20]))
            reward_record.append(reward_stableHead)

        reward_elbow_flair = 0
        if self.elbowFlairReward:
            root = self.robot_skeleton.bodynodes[1].to_world(np.zeros(3))
            spine = self.robot_skeleton.bodynodes[2].to_world(np.zeros(3))
            elbow = self.robot_skeleton.bodynodes[self.elbowFlairNode].to_world(np.zeros(3))
            dist = pyutils.distToLine(p=elbow, l0=root, l1=spine)
            z = 0.5
            s = 16
            l = 0.2
            reward_elbow_flair = -(1 - (z * math.tanh(s * (dist - l)) + z))
            # print("reward_elbow_flair: " + str(reward_elbow_flair))
            reward_record.append(reward_elbow_flair)

        reward_limbprogress = 0
        if self.limbProgressReward:
            if self.simulateCloth:
                #self.limbProgress = pyutils.limbFeatureProgress(
                #    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL,
                #                                      offset=self.fingertip), feature=self.sleeveLSeamFeature)

                self.limbProgress = pyutils.limbBoxProgress(limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL, offset=self.fingertip), boxFrame=self.rigidClothFrame)
                reward_limbprogress = self.limbProgress
                if reward_limbprogress < 0:  # remove euclidean distance penalty before containment
                    reward_limbprogress = 0
            reward_record.append(reward_limbprogress)

        avgContactGeodesic = None
        if self.numSteps > 0 and self.simulateCloth:
            contactInfo = pyutils.getContactIXGeoSide(sensorix=21, clothscene=self.clothScene,
                                                      meshgraph=self.separatedMesh)
            if len(contactInfo) > 0:
                avgContactGeodesic = 0
                for c in contactInfo:
                    avgContactGeodesic += c[1]
                avgContactGeodesic /= len(contactInfo)

        self.prevAvgGeodesic = avgContactGeodesic

        reward_oracleDisplacement = 0
        if self.oracleDisplacementReward:
            if np.linalg.norm(self.prevOracle) > 0 and self.localLeftEfShoulder1 is not None:
                # world_ef_displacement = wRFingertip2 - wRFingertip1
                relative_displacement = localLeftEfShoulder2 - self.localLeftEfShoulder1
                oracle0 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip2 + self.prevOracle) - localLeftEfShoulder2
                # oracle0 = oracle0/np.linalg.norm(oracle0)
                reward_oracleDisplacement += relative_displacement.dot(oracle0)
            reward_record.append(reward_oracleDisplacement)

        reward_contactGeo = 0
        if self.contactGeoReward:
            if self.simulateCloth:
                if self.limbProgress > 0:
                    reward_contactGeo = 1.0
                elif avgContactGeodesic is not None:
                    reward_contactGeo = 1.0 - (avgContactGeodesic / self.separatedMesh.maxGeo)
                    # reward_contactGeo = 1.0 - minContactGeodesic / self.separatedMesh.maxGeo
            reward_record.append(reward_contactGeo)

        clothDeformation = 0
        if self.simulateCloth:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
            self.deformation = clothDeformation

        reward_clothdeformation = 0
        if self.deformationPenalty:
            # reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_clothdeformation = -(math.tanh(
                0.14 * (clothDeformation - 25)) + 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_record.append(reward_clothdeformation)
        self.previousDeformationReward = reward_clothdeformation
        # force magnitude penalty
        reward_ctrl = -np.square(tau).sum()

        reward_restPose = 0
        restPoseError = 0
        if self.restPoseReward:
            if self.restPose is not None:
                #z = 0.5  # half the max magnitude (e.g. 0.5 -> [0,1])
                #s = 1.0  # steepness (higher is steeper)
                #l = 4.2  # translation
                dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
                restPoseError = dist
                #reward_restPose = -(z * math.tanh(s * (dist - l)) + z)
                reward_restPose = max(-51, -dist)
            # print("distance: " + str(dist) + " -> " + str(reward_restPose))
            reward_record.append(reward_restPose)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        # update graphs
        if self.limbProgressGraphing and self.reset_number > 0:
            # print(self.reset_number-1)
            # print(len(self.limbProgressGraph.yData))
            self.limbProgressGraph.yData[self.reset_number - 1][self.numSteps] = self.limbProgress
            if self.numSteps % 5 == 0:
                self.limbProgressGraph.update()

        # update graphs
        if self.restPoseErrorGraphing and self.reset_number > 0:
            self.restPoseErrorGraph.yData[self.reset_number - 1][self.numSteps] = restPoseError
            if self.numSteps % 5 == 0:
                self.restPoseErrorGraph.update()

        self.reward = reward_ctrl * 0 \
                      + reward_upright * self.uprightRewardWeight\
                      + reward_stableHead * self.stableHeadRewardWeight \
                      + reward_limbprogress * self.limbProgressRewardWeight \
                      + reward_contactGeo * self.contactGeoRewardWeight \
                      + reward_clothdeformation * self.deformationPenaltyWeight \
                      + reward_oracleDisplacement * self.oracleDisplacementRewardWeight \
                      + reward_elbow_flair * self.elbowFlairRewardWeight \
                      + reward_restPose * self.restPoseRewardWeight
        return self.reward

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix, dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        obs = np.concatenate([np.cos(theta), np.sin(theta), dtheta]).ravel()

        if self.prevTauObs:
            obs = np.concatenate([obs, self.prevTau])

        if self.hapticsInObs:
            f = None
            if self.simulateCloth and self.hapticsAware:
                f = self.clothScene.getHapticSensorObs()#get force from simulation
            else:
                f = np.zeros(f_size)
            obs = np.concatenate([obs, f]).ravel()

        if self.featureInObs and self.simulateCloth:
            centroid = self.sleeveLMidFeature.plane.org

            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            disp = centroid-efL
            obs = np.concatenate([obs, centroid, disp]).ravel()

        if self.oracleInObs and self.simulateCloth:
            oracle = np.zeros(3)
            if self.reset_number == 0:
                a=0 #nothing
            elif self.limbProgress > 0:
                oracle = self.sleeveLSeamFeature.plane.normal
            else:
                minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=21,
                                                                                     clothscene=self.clothScene,
                                                                                     meshgraph=self.separatedMesh,
                                                                                     returnOnlyGeo=False)
                if minGeoVix is None:
                    #oracle points to the garment when ef not in contact
                    efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
                    #closeVert = self.clothScene.getCloseVertex(p=efR)
                    #target = self.clothScene.getVertexPos(vid=closeVert)

                    centroid = self.sleeveLMidFeature.plane.org

                    target = np.array(centroid)
                    vec = target - efL
                    oracle = vec/np.linalg.norm(vec)
                else:
                    vixSide = 0
                    if _side:
                        vixSide = 1
                    if minGeoVix >= 0:
                        oracle = self.separatedMesh.geoVectorAt(minGeoVix, side=vixSide)
            self.prevOracle = np.array(oracle)
            obs = np.concatenate([obs, oracle]).ravel()

        if self.contactIDInObs:
            HSIDs = self.clothScene.getHapticSensorContactIDs()
            obs = np.concatenate([obs, HSIDs]).ravel()

        return obs

    def additionalResets(self):
        if(self.reset_number > 0):
            print("ef_accuracy_info: " + str(self.ef_accuracy_info))
        self.ef_accuracy_info = {'best': 0, 'worst': 0, 'total': 0, 'average': 0}

        if self.limbProgressGraphing:
            #print("here!")
            self.limbProgressGraph.save("limbProgressGraph", "limbProgressGraphData")
            self.limbProgressGraph.xdata = np.arange(250)
            self.limbProgressGraph.plotData(ydata=np.zeros(250))
            self.limbProgressGraph.update()

        if self.restPoseErrorGraphing:
            self.restPoseErrorGraph.save("restPoseErrorGraph", "restPoseErrorGraphData")
            self.restPoseErrorGraph.xdata = np.arange(250)
            self.restPoseErrorGraph.plotData(ydata=np.zeros(250))
            self.restPoseErrorGraph.update()

        if self.graphSPDError:
            self.SPDErrorGraph.close()
            self.SPDErrorGraph = pyutils.LineGrapher(title="SPD Error Violation", numPlots=7, legend=True)
            for i in range(len(self.SPDErrorGraph.labels)):
                self.SPDErrorGraph.labels[i] = str(i)

        #if self.reset_number == 10:
        #    exit()

        #do any additional resetting here
        self.handFirst = False
        #print(self.robot_skeleton.bodynodes[9].to_world(np.zeros(3)))

        if self.simulateCloth and self.linearTrackActive:
            self.clothScene.translateCloth(0, np.array([-0.155, -0.1, 0.285]))
            #draw an initial location
            randoms = np.random.rand(6)

            '''#scripted 4 corners
            if self.reset_number == 0:
                randoms = np.zeros(6)
            elif self.reset_number == 1:
                randoms = np.array([0,0,0,0,1,0])
            elif self.reset_number == 2:
                randoms = np.array([0, 0, 0, 1, 1, 0])
            elif self.reset_number == 3:
                randoms = np.array([0, 0, 0, 1, 0, 0])
            else:
                exit()'''

            self.linearTrackTarget = np.array([
                LERP(self.trackEndRange[0][0], self.trackEndRange[1][0], randoms[0]),
                LERP(self.trackEndRange[0][1], self.trackEndRange[1][1], randoms[1]),
                LERP(self.trackEndRange[0][2], self.trackEndRange[1][2], randoms[2]),
            ])
            self.linearTrackOrigin = np.array([
                LERP(self.trackInitialRange[0][0], self.trackInitialRange[1][0], randoms[3]),
                LERP(self.trackInitialRange[0][1], self.trackInitialRange[1][1], randoms[4]),
                LERP(self.trackInitialRange[0][2], self.trackInitialRange[1][2], randoms[5]),
            ])
            self.clothScene.translateCloth(0, self.linearTrackOrigin)
            a=0

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        #qpos[16] = 1.9
        #qpos[1] = -0.5
        '''qpos = np.array(
            [-0.0483053659505, 0.0321213273351, 0.0173036909392, 0.00486290205677, -0.00284350018845, -0.634602301004,
             -0.359172622713, 0.0792754054027, 2.66867203095, 0.00489456931428, 0.000476966442889, 0.0234663491334,
             -0.0254520098678, 0.172782859361, -1.31351102137, 0.702315566312, 1.73993331669, -0.0422811572637,
             0.586669332152, -0.0122329947565, 0.00179736869435, -8.0625896949e-05])
        '''
        self.set_state(qpos, qvel)
        #self.loadCharacterState(filename="characterState_1starmin")
        self.restPose = qpos

        #self.sawyer_skel.set_velocities(self.np_random.uniform(low=-3.5, high=3.5, size=self.sawyer_skel.ndofs))
        sawyer_pose = np.array(self.sawyer_skel.q)
        sawyer_pose[:6] = np.array(self.sawyer_root_dofs)
        sawyer_pose[6:] = np.array(self.sawyer_rest)
        self.sawyer_skel.set_positions(sawyer_pose)
        T = self.sawyer_skel.bodynodes[0].world_transform()
        tempFrame = pyutils.ShapeFrame()
        tempFrame.setTransform(T)
        root_quat = tempFrame.quat
        root_quat = (root_quat.x, root_quat.y, root_quat.z, root_quat.w)

        p.resetBasePositionAndOrientation(self.pyBulletSawyer, posObj=np.zeros(3), ornObj=root_quat)
        self.setPosePyBullet(self.sawyer_skel.q[6:])


        if(self.trackPosePath):
            a=0
            self.posePath = pyutils.Spline()
            pos_upper_lim = self.sawyer_skel.position_upper_limits()
            pos_lower_lim = self.sawyer_skel.position_lower_limits()
            for i in range(3):
                #pick a valid pose
                pose = np.zeros(7)
                for d in range(7):
                    ulim = pos_upper_lim[d+6]
                    llim = pos_lower_lim[d+6]
                    pose[d] = (random.random() * (ulim-llim)) + llim
                self.posePath.insert(p=pose, t=i*0.5)

            self.checkPoseSplineValidity()

            #check
            #for po in self.posePath.points:
            #    print(po.t)
            #    print(po.p)

            self.sawyer_skel.set_velocities(np.zeros(len(self.sawyer_skel.dq)))
            self.sawyer_skel.set_positions(np.concatenate([np.array(self.sawyer_root_dofs), self.posePath.points[0].p]))

        else: #setup IK path instead
            self.ikPath = pyutils.Spline()
            org = self.sawyer_skel.bodynodes[3].to_world(np.zeros(3))
            #spherical rejection sampling in reach range
            rands = []
            tarRange = self.maxSawyerReach*0.9
            for i in range(3):
                rands.append(np.random.uniform(-tarRange, tarRange, size=(3,)))
                while(np.linalg.norm(rands[i]) > 1 or rands[i][2] < 0):
                    rands[i] = np.random.uniform(-tarRange, tarRange, size=(3,))
                    rands[i][2] = abs(rands[i][2])
                self.ikPath.insert(t=0.5*i, p=org+rands[i])

            self.checkIKSplineValidity()
            self.ikTarget = self.ikPath.points[0].p

            self.rigidClothTargetFrame.setFromDirectionandUp(dir=-self.ikTarget, up=np.array([0, -1.0, 0]),
                                                             org=self.ikTarget)
            tar_quat = self.rigidClothTargetFrame.quat
            tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
            tar_dir = -self.ikTarget / np.linalg.norm(self.ikTarget)

            result = None
            if (self.ikOrientation):
                result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletSawyer,
                                                      endEffectorLinkIndex=12,
                                                      targetPosition=self.ikTarget - self.sawyer_root_dofs[3:],
                                                      targetOrientation=tar_quat,
                                                      # targetOrientation=tar_dir,
                                                      lowerLimits=self.sawyer_dof_llim.tolist(),
                                                      upperLimits=self.sawyer_dof_ulim.tolist(),
                                                      jointRanges=self.sawyer_dof_jr.tolist(),
                                                      restPoses=self.sawyer_skel.q[6:].tolist()
                                                      )
            else:
                result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletSawyer,
                                                      endEffectorLinkIndex=12,
                                                      targetPosition=self.ikTarget - self.sawyer_root_dofs[3:],
                                                      # targetOrientation=tar_quat,
                                                      # targetOrientation=tar_dir,
                                                      lowerLimits=self.sawyer_dof_llim.tolist(),
                                                      upperLimits=self.sawyer_dof_ulim.tolist(),
                                                      jointRanges=self.sawyer_dof_jr.tolist(),
                                                      restPoses=self.sawyer_skel.q[6:].tolist()
                                                      )


            self.setPosePyBullet(result)
            self.sawyer_skel.set_velocities(np.zeros(len(self.sawyer_skel.dq)))
            self.sawyer_skel.set_positions(np.concatenate([np.array(self.sawyer_root_dofs), result]))

            hn = self.sawyer_skel.bodynodes[14]  # hoop 1 node
            self.rigidClothFrame.setTransform(hn.world_transform())

            ef_accuracy = np.linalg.norm(self.sawyer_skel.bodynodes[13].to_world(np.zeros(3)) - self.ikTarget)
            retry_count = 0
            while(ef_accuracy > 0.05 and retry_count < 10):
                retry_count += 1
                print("retry " + str(retry_count))
                if (self.ikOrientation):
                    result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletSawyer,
                                                          endEffectorLinkIndex=12,
                                                          targetPosition=self.ikTarget - self.sawyer_root_dofs[3:],
                                                          targetOrientation=tar_quat,
                                                          # targetOrientation=tar_dir,
                                                          lowerLimits=self.sawyer_dof_llim.tolist(),
                                                          upperLimits=self.sawyer_dof_ulim.tolist(),
                                                          jointRanges=self.sawyer_dof_jr.tolist(),
                                                          restPoses=self.sawyer_skel.q[6:].tolist()
                                                          )
                else:
                    result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletSawyer,
                                                          endEffectorLinkIndex=12,
                                                          targetPosition=self.ikTarget - self.sawyer_root_dofs[3:],
                                                          # targetOrientation=tar_quat,
                                                          # targetOrientation=tar_dir,
                                                          lowerLimits=self.sawyer_dof_llim.tolist(),
                                                          upperLimits=self.sawyer_dof_ulim.tolist(),
                                                          jointRanges=self.sawyer_dof_jr.tolist(),
                                                          restPoses=self.sawyer_skel.q[6:].tolist()
                                                          )

                self.setPosePyBullet(result)
                self.sawyer_skel.set_positions(np.concatenate([np.array(self.sawyer_root_dofs), result]))
                ef_accuracy = np.linalg.norm(self.sawyer_skel.bodynodes[13].to_world(np.zeros(3)) - self.ikTarget)
            #DONE: IK setup

        if self.handleNode is not None:
            self.handleNode.clearHandles()
            #self.handleNode.addVertices(verts=[727, 138, 728, 1361, 730, 961, 1213, 137, 724, 1212, 726, 960, 964, 729, 155, 772])
            self.handleNode.addVertices(verts=[1552, 2090, 1525, 954, 1800, 663, 1381, 1527, 1858, 1077, 759, 533, 1429, 1131])
            self.handleNode.setOrgToCentroid()
            #if self.updateHandleNodeFrom >= 0:
            #    self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.recomputeOffsets()

        if self.simulateCloth:
            if self.sleeveLSeamFeature is not None:
                self.sleeveLSeamFeature.fitPlane(normhint=np.array([1.0, 0, 0]))
            if self.sleeveLEndFeature is not None:
                self.sleeveLEndFeature.fitPlane()
            if self.sleeveLEndFeature is not None:
                self.sleeveLMidFeature.fitPlane()

            #confirm relative normals
            # ensure relative correctness of normals
            CP2_CP1 = self.sleeveLEndFeature.plane.org - self.sleeveLMidFeature.plane.org
            CP2_CP0 = self.sleeveLSeamFeature.plane.org - self.sleeveLMidFeature.plane.org

            # if CP2 normal is not facing the sleeve end invert it
            if CP2_CP1.dot(self.sleeveLMidFeature.plane.normal) < 0:
                self.sleeveLMidFeature.plane.normal *= -1.0

            # if CP1 normal is facing the sleeve middle invert it
            if CP2_CP1.dot(self.sleeveLEndFeature.plane.normal) < 0:
                self.sleeveLEndFeature.plane.normal *= -1.0

            # if CP0 normal is not facing sleeve middle invert it
            if CP2_CP0.dot(self.sleeveLSeamFeature.plane.normal) > 0:
                self.sleeveLSeamFeature.plane.normal *= -1.0

            if self.reset_number == 0:
                self.separatedMesh.initSeparatedMeshGraph()
                self.separatedMesh.updateWeights()
                self.separatedMesh.computeGeodesic(feature=self.sleeveLMidFeature, oneSided=True, side=0, normalSide=1)

            if self.limbProgressReward:
                self.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL, offset=self.fingertip), feature=self.sleeveLSeamFeature)

        a=0

    def extraRenderFunction(self):
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        if(not self.trackPosePath):#draw IK
            self.ikPath.draw()
            renderUtils.setColor(color=[1.0, 0, 0])
            renderUtils.drawSphere(self.ikTarget)
            renderUtils.setColor(color=[0, 1.0, 0])
            renderUtils.drawLines(lines=[[np.zeros(3), self.sawyer_skel.bodynodes[3].to_world(np.zeros(3))]])
            renderUtils.drawSphere(self.sawyer_skel.bodynodes[13].to_world(np.zeros(3)))

        #render sawyer reach
        if self.renderSawyerReach:
            renderUtils.setColor(color=[0.75, 0.75, 0.75])
            renderUtils.drawSphere(pos=self.sawyer_skel.bodynodes[3].to_world(np.zeros(3)), rad=self.maxSawyerReach, solid=False)

        #render rigid cloth frame
        #test the intersection codes
        #tp0 = np.zeros(3)
        #tp1 = np.array([0.1, 0.4, -0.5])
        #tp1 /= np.linalg.norm(tp1)
        #renderUtils.drawArrow(p0=tp0, p1=tp1)
        #if(self.rigidClothFrame.intersects(_p=tp0, _v=tp1)[0]):
        #    self.rigidClothFrame.draw(fill=True)
        self.rigidClothFrame.draw(fill=False)
        self.rigidClothFrame.drawFrame()
        self.rigidClothTargetFrame.draw()
        self.rigidClothTargetFrame.drawFrame()
        #renderUtils.drawLines(lines=[[self.rigidClothFrame.org, np.zeros(3)]])
        #hn = self.sawyer_skel.bodynodes[13] #hand node
        #p0 = hn.to_world(np.zeros(3))
        #px = hn.to_world(np.array([1.0,0,0]))
        #py = hn.to_world(np.array([0,1.0,0]))
        #pz = hn.to_world(np.array([0,0,1.0]))
        #renderUtils.setColor(color=[1.0,0,0])
        #renderUtils.drawArrow(p0=p0, p1=px)
        #renderUtils.setColor(color=[0,1.0,0])
        #renderUtils.drawArrow(p0=p0, p1=py)
        #renderUtils.setColor(color=[0,0,1.0])
        #renderUtils.drawArrow(p0=p0, p1=pz)

        #draw pybullet sawyer body positions
        if False:
            for i in range(13):
                #print(p.getLinkState(self.pyBulletSawyer, i))
                pybullet_state = p.getLinkState(self.pyBulletSawyer, i)[0]
                renderUtils.setColor(color=[0, 0.0, 0])
                renderUtils.drawSphere(pybullet_state)

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        renderUtils.drawLineStrip(points=[
                                        self.robot_skeleton.bodynodes[12].to_world(self.fingertip),
                                        self.prevOracle+self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
                                          ])

        renderUtils.drawBox(cen=self.sawyer_root_dofs[3:], dim=np.array([0.2, 0.05, 0.2]))

        if(self.renderCloth):
            if self.sleeveLSeamFeature is not None:
                self.sleeveLSeamFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
            if self.sleeveLEndFeature is not None:
                self.sleeveLEndFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
            if self.sleeveLMidFeature is not None:
                self.sleeveLMidFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)

        #draw the linear track initial and end boxes
        if self.linearTrackActive:
            originCentroid = (self.trackInitialRange[0] + self.trackInitialRange[1])/2.0
            endCentroid = (self.trackEndRange[0] + self.trackEndRange[1])/2.0
            originDim = self.trackInitialRange[1] - self.trackInitialRange[0]
            endDim = self.trackEndRange[1] - self.trackEndRange[0]
            renderUtils.drawBox(cen=originCentroid, dim=originDim, fill=False)
            renderUtils.drawBox(cen=endCentroid, dim=endDim, fill=False)
            renderUtils.drawLines(lines=[[self.linearTrackOrigin, self.linearTrackTarget]])

        # render geodesic
        if False:
            for v in range(self.clothScene.getNumVertices()):
                side1geo = self.separatedMesh.nodes[v + self.separatedMesh.numv].geodesic
                side0geo = self.separatedMesh.nodes[v].geodesic

                pos = self.clothScene.getVertexPos(vid=v)
                norm = self.clothScene.getVertNormal(vid=v)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo, value=self.separatedMesh.maxGeo-side0geo))
                renderUtils.drawSphere(pos=pos-norm*0.01, rad=0.01)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo, value=self.separatedMesh.maxGeo-side1geo))
                renderUtils.drawSphere(pos=pos + norm * 0.01, rad=0.01)


        m_viewport = self.viewer.viewport
        # print(m_viewport)

        if self.renderUI:
            if self.renderRewardsData:
                self.rewardsData.render(topLeft=[m_viewport[2] - 410, m_viewport[3] - 15],
                                        dimensions=[400, -m_viewport[3] + 30])

            textHeight = 15
            textLines = 2

            renderUtils.setColor(color=[0.,0,0])
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Seed = " + str(self.setSeed), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps) + ", dt = " + str(self.dt) + ", frameskip = " + str(self.frame_skip), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Time = " + str(self.numSteps*self.dt), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Reward = " + str(self.reward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Previous Avg Geodesic = " + str(self.prevAvgGeodesic), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Limb Progress = " + str(self.limbProgress), color=(0., 0, 0))
            textLines += 1

            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False)

            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=self.limbProgress, color=[0.0, 3.0, 0])
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 30], h=16, w=60, progress=-self.previousDeformationReward, color=[1.0, 0.0, 0])

            #draw Sawyer positions vs. limits
            for d in range(7):
                self.clothScene.drawText(x=15., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.sawyer_skel.dofs[6+d].position_lower_limit(),), color=(0., 0, 0))
                self.clothScene.drawText(x=100., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.sawyer_skel.q[6+d],), color=(0., 0, 0))
                self.clothScene.drawText(x=200., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.sawyer_skel.dofs[6+d].position_upper_limit(),), color=(0., 0, 0))

                val = (self.sawyer_skel.q[6+d] - self.sawyer_skel.dofs[6+d].position_lower_limit())/(self.sawyer_skel.dofs[6+d].position_upper_limit()-self.sawyer_skel.dofs[6+d].position_lower_limit())
                tar = (self.previousIKResult[d] - self.sawyer_skel.dofs[6+d].position_lower_limit())/(self.sawyer_skel.dofs[6+d].position_upper_limit()-self.sawyer_skel.dofs[6+d].position_lower_limit())
                renderUtils.drawProgressBar(topLeft=[75, self.viewer.viewport[3] - 450 - d*20], h=16, w=120, progress=val, origin=0.5, features=[tar], color=[1.0, 0.0, 0])


                self.clothScene.drawText(x=250., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.sawyer_skel.force_lower_limits()[6+d],), color=(0., 0, 0))
                self.clothScene.drawText(x=335., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.sawyer_skel.forces()[6+d],), color=(0., 0, 0))
                self.clothScene.drawText(x=435., y=self.viewer.viewport[3] - 463 - d*20, text="%0.2f" % (self.sawyer_skel.force_upper_limits()[6+d],), color=(0., 0, 0))

                tval = (self.sawyer_skel.forces()[6+d]-self.sawyer_skel.force_lower_limits()[6+d])/(self.sawyer_skel.force_upper_limits()[6+d]-self.sawyer_skel.force_lower_limits()[6+d])
                renderUtils.drawProgressBar(topLeft=[310, self.viewer.viewport[3] - 450 - d * 20], h=16, w=120, progress=tval, origin=0.5, color=[1.0, 0.0, 0])

        # render target pose
        if self.viewer is not None and self.renderIKGhost and not self.trackPosePath:
            q = np.array(self.sawyer_skel.q)
            dq = np.array(self.sawyer_skel.dq)
            self.sawyer_skel.set_positions(np.concatenate([np.array(self.sawyer_root_dofs), self.previousIKResult]))
            # self.viewer.scene.render(self.viewer.sim)
            self.sawyer_skel.render()
            self.sawyer_skel.set_positions(q)
            self.sawyer_skel.set_velocities(dq)

        if self.viewer is not None and self.trackPosePath:
            q = np.array(self.sawyer_skel.q)
            dq = np.array(self.sawyer_skel.dq)
            samples = 100
            framefreq = 5
            ef_locations = []
            target_drawn=False
            for i in range(samples):
                t = (self.posePath.points[-1].t-self.posePath.points[0].t)*(i/(samples-1))
                #print(t)
                #print(self.posePath.pos(t=t))
                self.sawyer_skel.set_positions(np.concatenate([np.array(self.sawyer_root_dofs), self.posePath.pos(t=t)]))
                ef_frame = pyutils.BoxFrame(c0=np.array([0.1, 0.2, 0.001]), c1=np.array([-0.1, 0, -0.001]))
                hn = self.sawyer_skel.bodynodes[14]  # hoop 1 node
                ef_frame.setTransform(hn.world_transform())
                ef_locations.append(ef_frame.org)
                if(self.numSteps*self.ikPathTimeScale < t and not target_drawn):
                    target_drawn = True
                    renderUtils.setColor(color=[1.0, 0.0, 0.0])
                    renderUtils.drawSphere(pos=ef_frame.org)
                if(i%framefreq == 0):
                    ef_frame.drawFrame(size=0.2)
                    renderUtils.setColor(color=[0.5,0.5,0.5])
                    ef_frame.draw()

            #draw the ef_curve
            renderUtils.drawLineStrip(ef_locations)

                #self.sawyer_skel.render()
            self.sawyer_skel.set_positions(q)
            self.sawyer_skel.set_velocities(dq)

    def saveGripperState(self, filename=None):
        print("saving gripper state")
        if filename is None:
            filename = "gripperState"
        print("filename " + str(filename))
        f = open(filename, 'w')
        for ix, dof in enumerate(self.dart_world.skeletons[0].q):
            if ix > 0:
                f.write(" ")
            if ix < 3:
                f.write(str(self.handleNode.org[ix]))
            else:
                f.write(str(dof))

        f.write("\n")

        for ix, dof in enumerate(self.dart_world.skeletons[0].dq):
            if ix > 0:
                f.write(" ")
            f.write(str(dof))
        f.close()

    # set a pose in the pybullet simulation env
    def setPosePyBullet(self, pose):
        count = 0
        for i in range(p.getNumJoints(self.pyBulletSawyer)):
            jinfo = p.getJointInfo(self.pyBulletSawyer, i)
            if (jinfo[3] > -1):
                p.resetJointState(self.pyBulletSawyer, i, pose[count])
                count += 1

    def checkIKSplineValidity(self):
        steps = 1.0/self.ikPathTimeScale #number of steps to complete the path
        results = []
        for i in range(math.ceil(steps)):
            t = i*self.ikPathTimeScale
            ikTarget = self.ikPath.pos(t)
            rigidClothTargetFrame = pyutils.BoxFrame()
            rigidClothTargetFrame.setFromDirectionandUp(dir=-ikTarget, up=np.array([0, -1.0, 0]),org=ikTarget)
            tar_quat = rigidClothTargetFrame.quat
            tar_quat = (tar_quat.x, tar_quat.y, tar_quat.z, tar_quat.w)
            pose = []
            for i in range(p.getNumJoints(self.pyBulletSawyer)):
                jinfo = p.getJointInfo(self.pyBulletSawyer, i)
                if (jinfo[3] > -1):
                    pose.append(p.getJointState(self.pyBulletSawyer, i)[0])

            result = p.calculateInverseKinematics(bodyUniqueId=self.pyBulletSawyer,
                                                  endEffectorLinkIndex=12,
                                                  targetPosition=ikTarget - self.sawyer_root_dofs[3:],
                                                  targetOrientation=tar_quat,
                                                  lowerLimits=self.sawyer_dof_llim.tolist(),
                                                  upperLimits=self.sawyer_dof_ulim.tolist(),
                                                  jointRanges=self.sawyer_dof_jr.tolist(),
                                                  restPoses=pose
                                                  )
            results.append(np.array(result))
            self.setPosePyBullet(result)

        vels = []
        invalid_count = 0
        for r in range(1,len(results)):
            #print(results[r])
            vels.append((results[r]-results[r-1])/self.dt)
            for d in range(7):
                if(abs(vels[-1][d]) > self.sawyer_skel.dofs[d+6].velocity_upper_limit()):
                    invalid_count += 1
        print("Spline checked with " + str(invalid_count) + " invalid IK velocities.")
        self.setPosePyBullet(np.zeros(7))

    def checkPoseSplineValidity(self):
        steps = 1.0 / self.ikPathTimeScale  # number of steps to complete the path
        results = []
        for i in range(math.ceil(steps)):
            t = i * self.ikPathTimeScale
            results.append(self.posePath.pos(t=t))
        vels = []
        invalid_count = 0
        for r in range(1, len(results)):
            # print(results[r])
            vels.append((results[r] - results[r - 1]) / self.dt)
            for d in range(7):
                if (abs(vels[-1][d]) > self.sawyer_skel.dofs[d + 6].velocity_upper_limit()):
                    invalid_count += 1
        print("Spline checked with " + str(invalid_count) + " invalid pose velocities.")


def LERP(p0, p1, t):
    return p0 + (p1 - p0) * t
