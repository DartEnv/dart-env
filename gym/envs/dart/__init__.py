from gym.envs.dart.dart_env import DartEnv
# ^^^^^ so that user gets the correct error
# message if Dart is not installed correctly
from gym.envs.dart.parameter_managers import *

from gym.envs.dart.cart_pole import DartCartPoleEnv
from gym.envs.dart.hopper import DartHopperEnv
#from gym.envs.dart.full_body import DartFullbodyEnv
#from gym.envs.dart.hopperRBF import DartHopperRBFEnv
#from gym.envs.dart.hopper_cont import DartHopperEnvCont
from gym.envs.dart.reacher import DartReacherEnv
from gym.envs.dart.robot_walk import DartRobotWalk
from gym.envs.dart.cart_pole_img import DartCartPoleImgEnv

#cloth:
from gym.envs.dart.sphere_tube import DartClothSphereTubeEnv
from gym.envs.dart.reacher_cloth import DartClothReacherEnv
from gym.envs.dart.reacher_cloth_1arm import DartClothReacherEnv2
from gym.envs.dart.reacher_cloth_1arm_spline import DartClothReacherEnv3
from gym.envs.dart.reacher_cloth_sleeve import DartClothSleeveReacherEnv
from gym.envs.dart.reacher_cloth_shirt import DartClothShirtReacherEnv
from gym.envs.dart.posereacher_cloth import DartClothPoseReacherEnv
from gym.envs.dart.cloth_testbed import DartClothTestbedEnv
from gym.envs.dart.gown_dressing_demo import DartClothGownDemoEnv
from gym.envs.dart.gripped_tshirt_demo import DartClothGrippedTshirtEnv
from gym.envs.dart.gripped_tshirt_targetspline import DartClothGrippedTshirtSplineEnv
from gym.envs.dart.gripped_tshirt_targetspline_2ndarm import DartClothGrippedTshirtSpline2ndArmEnv
from gym.envs.dart.endeffectordisplacer import DartClothEndEffectorDisplacerEnv
from gym.envs.dart.jointlimitstest import DartClothJointLimitsTestEnv
from gym.envs.dart.upperbodydatadriven import DartClothUpperBodyDataDrivenEnv
from gym.envs.dart.upperbodydatadriven_tshirt import DartClothUpperBodyDataDrivenTshirtEnv
from gym.envs.dart.upperbodydatadriven_cloth_tshirtR import DartClothUpperBodyDataDrivenClothTshirtREnv
from gym.envs.dart.upperbodydatadriven_cloth_tshirtL import DartClothUpperBodyDataDrivenClothTshirtLEnv
from gym.envs.dart.upperbodydatadriven_cloth_reacher import DartClothUpperBodyDataDrivenClothReacherEnv
from gym.envs.dart.upperbodydatadriven_cloth_dropgrip import DartClothUpperBodyDataDrivenClothDropGripEnv
from gym.envs.dart.upperbodydatadriven_cloth_phaseinterpolate import DartClothUpperBodyDataDrivenClothPhaseInterpolateEnv
from gym.envs.dart.upperbodydatadriven_cloth_phaseinterpolate2 import DartClothUpperBodyDataDrivenClothPhaseInterpolate2Env

#multiagent
from gym.envs.dart.multiagent import DartMultiAgentEnv

from gym.envs.dart.walker2d import DartWalker2dEnv
from gym.envs.dart.walker3d import DartWalker3dEnv
from gym.envs.dart.walker3d_spd import DartWalker3dSPDEnv
from gym.envs.dart.inverted_double_pendulum import DartDoubleInvertedPendulumEnv
from gym.envs.dart.dog import DartDogEnv
from gym.envs.dart.reacher2d import DartReacher2dEnv

