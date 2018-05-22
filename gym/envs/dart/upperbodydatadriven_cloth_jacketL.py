# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.upperbodydatadriven_cloth_base import *
import random
import time
import math

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils
from pyPhysX.pyutils import LERP
import pyPhysX.renderUtils
import pyPhysX.meshgraph as meshgraph
from pyPhysX.clothfeature import *

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

class DartClothUpperBodyDataDrivenClothJacketLEnv(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = False
        clothSimulation = True
        renderCloth = True
        dt = 0.002
        frameskip = 5

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
        self.restPoseReward             = False

        self.uprightRewardWeight              = 1  #if true, rewarded for 0 torso angle from vertical
        self.stableHeadRewardWeight           = 1
        self.elbowFlairRewardWeight           = 1
        self.limbProgressRewardWeight         = 10  # if true, the (-inf, 1] plimb progress metric is included in reward
        self.oracleDisplacementRewardWeight   = 50  # if true, reward ef displacement in the oracle vector direction
        self.contactGeoRewardWeight           = 2  # if true, [0,1] reward for ef contact geo (0 if no contact, 1 if limbProgress > 0).
        self.deformationPenaltyWeight         = 5
        self.restPoseRewardWeight             = 0.05

        #other flags
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.collarTermination  = False  #if true, rollout terminates when collar is off the head/neck
        self.sleeveEndTerm      = True  #if true, terminate the rollout if the arm enters the end of sleeve feature before the beginning (backwards dressing)
        self.elbowFirstTerm     = True #if true, terminate when any limb enters the feature before the hand
        self.resetStateFromDistribution = True
        self.resetDistributionPrefix = "saved_control_states_jacket/enter_seq_lsleeve"
        self.resetDistributionSize = 20
        self.state_save_directory = "saved_control_states/"
        self.curriculumBias = 0 #chance that a curriculum initialization is used
        self.curriculumSet = [6,8]
        if self.curriculumBias > 0:
            print("Using a curriculum bias of " + str(self.curriculumBias))

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

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1280,720),
                                                          clothMeshFile="jacketmedium.obj",
                                                          #clothMeshStateFile = "startJacketSleeveL.obj",
                                                          #clothMeshStateFile = "objFile_1starmin.obj",
                                                          clothScale=np.array([0.7,0.7,0.5]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation,
                                                          dt=dt,
                                                          frameskip=frameskip)

        #clothing features
        self.sleeveLVerts = [848, 648, 649, 652, 980, 1380, 861, 860, 862, 1369, 92, 1344, 940, 941, 953, 561, 559, 560, 788, 789, 814, 1261, 537, 1122, 535, 1121, 536, 1277, 831, 1278, 834, 1287]
        self.sleeveLMidVerts = [1033, 1013, 1014, 1016, 1035, 1017, 1018, 1020, 1031, 629, 630, 633, 1028, 1030, 1037, 1025, 1026, 1021, 1022, 1414, 1024, 1422]
        self.sleeveLEndVerts = [1015, 216, 1036, 217, 1019, 218, 1032, 72, 632, 74, 222, 1038, 221, 1027, 219, 1023, 220, 1034, 215]
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveLEndFeature = ClothFeature(verts=self.sleeveLEndVerts, clothScene=self.clothScene)
        self.sleeveLMidFeature = ClothFeature(verts=self.sleeveLMidVerts, clothScene=self.clothScene)

        self.simulateCloth = clothSimulation
        if self.simulateCloth:
            self.handleNode = HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975]))

        if not renderCloth:
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

    def saveObjState(self):
        print("Trying to save the object state")
        self.clothScene.saveObjState("objState", 0)

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
        if self.handleNode is not None and self.numSteps < 1:
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.step()
        if self.numSteps > 0:
            self.handleNode.clearHandles()

        fingertip = np.array([0.0, -0.095, 0.0])
        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
        self.localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1)  # right fingertip in right shoulder local frame
        a=0

    def checkTermination(self, tau, s, obs):
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
        fingertip = np.array([0.0, -0.095, 0.0])
        wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
        wLFingertip2 = self.robot_skeleton.bodynodes[12].to_world(fingertip)
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
            reward_record.append(reward_elbow_flair)
            # print("reward_elbow_flair: " + str(reward_elbow_flair))

        reward_limbprogress = 0
        if self.limbProgressReward:
            if self.simulateCloth:
                self.limbProgress = pyutils.limbFeatureProgress(
                    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL,
                                                      offset=np.array([0, -0.095, 0])), feature=self.sleeveLSeamFeature)
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
            #reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_clothdeformation = -(math.tanh(0.14*(clothDeformation-25)) + 1)/2.0 # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_record.append(reward_clothdeformation)
        self.previousDeformationReward = reward_clothdeformation
        # force magnitude penalty
        reward_ctrl = -np.square(tau).sum()

        reward_restPose = 0
        if self.restPoseReward:
            if self.restPose is not None:
                z = 0.5  # half the max magnitude (e.g. 0.5 -> [0,1])
                s = 1.0  # steepness (higher is steeper)
                l = 4.2  # translation
                dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
                #reward_restPose = -(z * math.tanh(s * (dist - l)) + z)
                reward_restPose = max(-51, -dist)
                # print("distance: " + str(dist) + " -> " + str(reward_restPose))
            reward_record.append(reward_restPose)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        self.reward = reward_ctrl * 0 \
                      + reward_upright * self.uprightRewardWeight \
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

        fingertip = np.array([0.0, -0.095, 0.0])

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

            efL = self.robot_skeleton.bodynodes[12].to_world(fingertip)
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
                    efL = self.robot_skeleton.bodynodes[12].to_world(fingertip)
                    #closeVert = self.clothScene.getCloseVertex(p=efR)
                    #target = self.clothScene.getVertexPos(vid=closeVert)

                    centroid = self.sleeveLMidFeature.plane.org

                    target = centroid
                    vec = target - efL
                    oracle = vec/np.linalg.norm(vec)
                else:
                    vixSide = 0
                    if _side:
                        vixSide = 1
                    if minGeoVix >= 0:
                        oracle = self.separatedMesh.geoVectorAt(minGeoVix, side=vixSide)
            self.prevOracle = oracle
            obs = np.concatenate([obs, oracle]).ravel()

        if self.contactIDInObs:
            HSIDs = self.clothScene.getHapticSensorContactIDs()
            obs = np.concatenate([obs, HSIDs]).ravel()

        return obs

    def additionalResets(self):
        #do any additional resetting here
        self.handFirst = False
        #if self.simulateCloth:
        #    self.clothScene.translateCloth(0, np.array([0.125, -0.27, -0.6]))
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        #qpos =
        '''qpos = np.array(
            [-0.0483053659505, 0.0321213273351, 0.0173036909392, 0.00486290205677, -0.00284350018845, -0.634602301004,
             -0.359172622713, 0.0792754054027, 2.66867203095, 0.00489456931428, 0.000476966442889, 0.0234663491334,
             -0.0254520098678, 0.172782859361, -1.31351102137, 0.702315566312, 1.73993331669, -0.0422811572637,
             0.586669332152, -0.0122329947565, 0.00179736869435, -8.0625896949e-05])
        '''
        self.set_state(qpos, qvel)
        self.restPose = np.array(qpos)

        if self.resetStateFromDistribution:
            if self.reset_number == 0: #load the distribution
                count = 0
                objfname_ix = self.resetDistributionPrefix + "%05d" % count
                while os.path.isfile(objfname_ix + ".obj"):
                    count += 1
                    #print(objfname_ix)
                    self.clothScene.addResetStateFrom(filename=objfname_ix+".obj")
                    objfname_ix = self.resetDistributionPrefix + "%05d" % count

            resetStateNumber = -1
            if self.curriculumBias > 0:
                prob = random.random()
                if prob < self.curriculumBias:
                    resetStateNumber = random.randint(0, len(self.curriculumSet)-1)
                    resetStateNumber = self.curriculumSet[resetStateNumber]

            if resetStateNumber < 0:
                resetStateNumber = random.randint(0,self.resetDistributionSize-1)
            #resetStateNumber = 0
            #resetStateNumber = self.reset_number%self.resetDistributionSize
            #print(resetStateNumber)
            charfname_ix = self.resetDistributionPrefix + "_char%05d" % resetStateNumber
            self.clothScene.setResetState(cid=0, index=resetStateNumber)
            self.loadCharacterState(filename=charfname_ix)
            qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.1, high=0.1, size=self.robot_skeleton.ndofs)
            qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
            self.set_state(qpos, qvel)
        else:
            self.loadCharacterState(filename="characterState_startJacketSleeveL")

        if self.handleNode is not None:
            self.handleNode.clearHandles()
            self.handleNode.addVertices(verts=[727, 138, 728, 1361, 730, 961, 1213, 137, 724, 1212, 726, 960, 964, 729, 155, 772])
            self.handleNode.setOrgToCentroid()
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.recomputeOffsets()

        if self.simulateCloth:
            self.sleeveLSeamFeature.fitPlane(normhint=np.array([-1.0, 0, 0]))
            self.sleeveLEndFeature.fitPlane()
            self.sleeveLMidFeature.fitPlane()

            # confirm relative normals
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
                self.separatedMesh.computeGeodesic(feature=self.sleeveLMidFeature, oneSided=True, side=0, normalSide=0)

            if self.limbProgressReward:
                self.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL,offset=np.array([0,-0.095,0])), feature=self.sleeveLSeamFeature)

        a=0

    def extraRenderFunction(self):
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[12].to_world(np.array([0.0, -0.075, 0.0])),
                                          self.prevOracle+self.robot_skeleton.bodynodes[12].to_world(np.array([0.0, -0.075, 0.0])),
                                          ])

        if self.sleeveLSeamFeature is not None:
            self.sleeveLSeamFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
        if self.sleeveLEndFeature is not None:
            self.sleeveLEndFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
        if self.sleeveLMidFeature is not None:
            self.sleeveLMidFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)

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

        links = pyutils.getRobotLinks(self.robot_skeleton, pose=self.restPose)
        renderUtils.drawLines(lines=links)

        m_viewport = self.viewer.viewport
        # print(m_viewport)
        self.rewardsData.render(topLeft=[m_viewport[2] - 410, m_viewport[3] - 15],
                                dimensions=[400, -m_viewport[3] + 30])

        textHeight = 15
        textLines = 2

        if self.renderUI:
            renderUtils.setColor(color=[0.,0,0])
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps), color=(0., 0, 0))
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
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.restPose, renderRestPose=True)

            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=self.limbProgress, color=[0.0, 3.0, 0])
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 30], h=16, w=60, progress=-self.previousDeformationReward, color=[1.0, 0.0, 0])
