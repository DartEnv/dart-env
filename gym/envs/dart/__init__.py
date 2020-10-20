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
from gym.envs.dart.walker3d_full import DartWalker3dFullEnv
from gym.envs.dart.walker3d_restricted import DartWalker3dRestrictedEnv
from gym.envs.dart.walker3d_project import DartWalker3dProjectionEnv

from gym.envs.dart.walker3d_spd import DartWalker3dSPDEnv
from gym.envs.dart.inverted_double_pendulum import DartDoubleInvertedPendulumEnv
from gym.envs.dart.dog import DartDogEnv
from gym.envs.dart.dog_robot import DartDogRobotEnv
from gym.envs.dart.reacher2d import DartReacher2dEnv

from gym.envs.dart.cartpole_swingup import DartCartPoleSwingUpEnv

from gym.envs.dart.pendulum import DartPendulumEnv

from gym.envs.dart.walker2d_pendulum import DartWalker2dPendulumEnv

from gym.envs.dart.ball_walker import DartBallWalkerEnv

from gym.envs.dart.human_walker import DartHumanWalkerEnv
from gym.envs.dart.human_ballwalker import DartHumanBallWalkerEnv
from gym.envs.dart.human_balance import DartHumanBalanceEnv
from gym.envs.dart.humanoid_balance import DartHumanoidBalanceEnv

from gym.envs.dart.hopper_rss import DartHopperRSSEnv

from gym.envs.dart.hexapod import DartHexapodEnv

from gym.envs.dart.hopper_hybridcontrol import DartHopperHybridEnv

from gym.envs.dart.hopper_1link import DartHopper1LinkEnv
from gym.envs.dart.hopper_3link import DartHopper3LinkEnv
from gym.envs.dart.hopper_4link import DartHopper4LinkEnv
from gym.envs.dart.hopper_5link import DartHopper5LinkEnv
from gym.envs.dart.hopper_6link import DartHopper6LinkEnv
from gym.envs.dart.hopper_7link import DartHopper7LinkEnv
from gym.envs.dart.hopper_8link import DartHopper8LinkEnv

from gym.envs.dart.hopper_3link_2foot import DartHopper3Link2FootEnv
from gym.envs.dart.hopper_4link_2foot import DartHopper4Link2FootEnv
from gym.envs.dart.hopper_5link_2foot import DartHopper5Link2FootEnv
from gym.envs.dart.hopper_6link_2foot import DartHopper6Link2FootEnv

from gym.envs.dart.cartpole_1pole import DartCartPole1PoleEnv
from gym.envs.dart.cartpole_2pole import DartCartPole2PoleEnv
from gym.envs.dart.cartpole_3pole import DartCartPole3PoleEnv
from gym.envs.dart.cartpole_4pole import DartCartPole4PoleEnv
from gym.envs.dart.cartpole_5pole import DartCartPole5PoleEnv

from gym.envs.dart.snake_3link import DartSnake3LinkEnv
from gym.envs.dart.snake_4link import DartSnake4LinkEnv
from gym.envs.dart.snake_5link import DartSnake5LinkEnv
from gym.envs.dart.snake_6link import DartSnake6LinkEnv
from gym.envs.dart.snake_7link import DartSnake7LinkEnv
from gym.envs.dart.snake_8link import DartSnake8LinkEnv
from gym.envs.dart.snake_9link import DartSnake9LinkEnv
from gym.envs.dart.snake_10link import DartSnake10LinkEnv

from gym.envs.dart.hopper_4link_spd import DartHopper4LinkSPDEnv
from gym.envs.dart.hopper_5link_spd import DartHopper5LinkSPDEnv

from gym.envs.dart.reacher_2link import DartReacher2LinkEnv
from gym.envs.dart.reacher_3link import DartReacher3LinkEnv
from gym.envs.dart.reacher_4link import DartReacher4LinkEnv
from gym.envs.dart.reacher_5link import DartReacher5LinkEnv
from gym.envs.dart.reacher_6link import DartReacher6LinkEnv
from gym.envs.dart.reacher_7link import DartReacher7LinkEnv

from gym.envs.dart.ant import DartAntEnv

from gym.envs.dart.darwin import DartDarwinEnv
from gym.envs.dart.darwin_squat import DartDarwinSquatEnv
from gym.envs.dart.darwin_newfoot import DartDarwinNewFootEnv

from gym.envs.dart.robotis_op3 import DartOP3Env

from gym.envs.dart.halfcheetah import DartHalfCheetahEnv

from gym.envs.dart.hopper_soft import DartHopperSoftEnv

from gym.envs.dart.ddp_env import DDPEnv

from gym.envs.dart.cross_puzzle import DartCrossPuzzle
from gym.envs.dart.bridge_puzzle import DartBridgePuzzle
from gym.envs.dart.ball_circle import DartBallCircle

from gym.envs.dart.A1 import DartA1Env
from gym.envs.dart.franka_transport import DartFrankaTransportEnv