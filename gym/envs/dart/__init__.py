from gym.envs.dart.dart_env import DartEnv
# ^^^^^ so that user gets the correct error
# message if Dart is not installed correctly
from gym.envs.dart.parameter_managers import *

from gym.envs.dart.cart_pole import DartCartPoleEnv
from gym.envs.dart.hopper import DartHopperEnv
from gym.envs.dart.hopper_assist import DartHopperAssistEnv
from gym.envs.dart.hopper_backpack import DartHopperBackPackEnv
#from gym.envs.dart.hopperRBF import DartHopperRBFEnv
from gym.envs.dart.hopper_cont import DartHopperEnvCont
from gym.envs.dart.reacher import DartReacherEnv
from gym.envs.dart.manipulator2d import DartManipulator2dEnv
from gym.envs.dart.robot_walk import DartRobotWalk
from gym.envs.dart.cart_pole_img import DartCartPoleImgEnv
from gym.envs.dart.walker2d import DartWalker2dEnv
from gym.envs.dart.walker2d_backpack import DartWalker2dBackpackEnv
from gym.envs.dart.walker3d import DartWalker3dEnv
from gym.envs.dart.walker3d_restricted import DartWalker3dRestrictedEnv
from gym.envs.dart.walker3d_project import DartWalker3dProjectionEnv

from gym.envs.dart.walker3d_spd import DartWalker3dSPDEnv
from gym.envs.dart.inverted_double_pendulum import DartDoubleInvertedPendulumEnv
from gym.envs.dart.dog import DartDogEnv
from gym.envs.dart.reacher2d import DartReacher2dEnv

from gym.envs.dart.cartpole_swingup import DartCartPoleSwingUpEnv

from gym.envs.dart.walker2d_pendulum import DartWalker2dPendulumEnv

from gym.envs.dart.ball_walker import DartBallWalkerEnv

from gym.envs.dart.human_walker import DartHumanWalkerEnv

from gym.envs.dart.hopper_rss import DartHopperRSSEnv