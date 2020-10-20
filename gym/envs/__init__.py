from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------

register(
    id='Copy-v0',
    entry_point='gym.envs.algorithmic:CopyEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='RepeatCopy-v0',
    entry_point='gym.envs.algorithmic:RepeatCopyEnv',
    max_episode_steps=200,
    reward_threshold=75.0,
)

register(
    id='ReversedAddition-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 2},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='ReversedAddition3-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 3},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='DuplicatedInput-v0',
    entry_point='gym.envs.algorithmic:DuplicatedInputEnv',
    max_episode_steps=200,
    reward_threshold=9.0,
)

register(
    id='Reverse-v0',
    entry_point='gym.envs.algorithmic:ReverseEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

# Classic
# ----------------------------------------

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPole-v1',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='MountainCarContinuous-v0',
    entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='Acrobot-v1',
    entry_point='gym.envs.classic_control:AcrobotEnv',
    reward_threshold=-100.0,
    max_episode_steps=500,
)

# Box2d
# ----------------------------------------

register(
    id='LunarLander-v2',
    entry_point='gym.envs.box2d:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuous-v2',
    entry_point='gym.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='BipedalWalker-v3',
    entry_point='gym.envs.box2d:BipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id='BipedalWalkerHardcore-v3',
    entry_point='gym.envs.box2d:BipedalWalkerHardcore',
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id='CarRacing-v0',
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id='Blackjack-v0',
    entry_point='gym.envs.toy_text:BlackjackEnv',
)

register(
    id='KellyCoinflip-v0',
    entry_point='gym.envs.toy_text:KellyCoinflipEnv',
    reward_threshold=246.61,
)
register(
    id='KellyCoinflipGeneralized-v0',
    entry_point='gym.envs.toy_text:KellyCoinflipGeneralizedEnv',
)

register(
    id='FrozenLake-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4'},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

register(
    id='FrozenLake8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8'},
    max_episode_steps=200,
    reward_threshold=0.99, # optimum = 1
)

register(
    id='CliffWalking-v0',
    entry_point='gym.envs.toy_text:CliffWalkingEnv',
)

register(
    id='NChain-v0',
    entry_point='gym.envs.toy_text:NChainEnv',
    max_episode_steps=1000,
)

register(
    id='Roulette-v0',
    entry_point='gym.envs.toy_text:RouletteEnv',
    max_episode_steps=100,
)

register(
    id='Taxi-v3',
    entry_point='gym.envs.toy_text:TaxiEnv',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=200,
)

register(
    id='GuessingGame-v0',
    entry_point='gym.envs.toy_text:GuessingGame',
    max_episode_steps=200,
)

register(
    id='HotterColder-v0',
    entry_point='gym.envs.toy_text:HotterColder',
    max_episode_steps=200,
)

# Point Reach
register(
    id='PointReach-v1',
    entry_point = 'gym.envs.classic_control:PointReachEnv',
    max_episode_steps = 500,
)

# Point Mass
register(
    id='PointMass-v1',
    entry_point = 'gym.envs.classic_control:PointMassEnv',
    max_episode_steps = 500,
)

# slit navigation
register(
    id='SlitNav-v1',
    entry_point = 'gym.envs.classic_control:Slit_Navigation',
    max_episode_steps = 400,
)

register(
    id='MiniCar-v1',
    entry_point = 'gym.envs.classic_control:MiniCarEnv',
    max_episode_steps = 1000,
)

# LQR Env
register(
    id='LQR-v1',
    entry_point = 'gym.envs.classic_control:LQREnv',
    max_episode_steps = 500,
)

# Dart
# ----------------------------------------
register(
    id='DartCartPole1Pole-v1',
    entry_point='gym.envs.dart:DartCartPole1PoleEnv',
    reward_threshold=950.0,
    max_episode_steps=1000,
)

register(
    id='DartCartPole2Pole-v1',
    entry_point='gym.envs.dart:DartCartPole2PoleEnv',
    reward_threshold=950.0,
    max_episode_steps=1000,
)

register(
    id='DartCartPole3Pole-v1',
    entry_point='gym.envs.dart:DartCartPole3PoleEnv',
    reward_threshold=950.0,
    max_episode_steps=1000,
)

register(
    id='DartCartPole4Pole-v1',
    entry_point='gym.envs.dart:DartCartPole4PoleEnv',
    reward_threshold=950.0,
    max_episode_steps=1000,
)

register(
    id='DartCartPole5Pole-v1',
    entry_point='gym.envs.dart:DartCartPole5PoleEnv',
    reward_threshold=950.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper-v1',
    entry_point='gym.envs.dart:DartHopperEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopperHybrid-v1',
    entry_point='gym.envs.dart:DartHopperHybridEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopperSoft-v1',
    entry_point='gym.envs.dart:DartHopperSoftEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper1Link-v1',
    entry_point='gym.envs.dart:DartHopper1LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=100,
)

register(
    id='DartHopper3Link-v1',
    entry_point='gym.envs.dart:DartHopper3LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper4Link-v1',
    entry_point='gym.envs.dart:DartHopper4LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper5Link-v1',
    entry_point='gym.envs.dart:DartHopper5LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper6Link-v1',
    entry_point='gym.envs.dart:DartHopper6LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper7Link-v1',
    entry_point='gym.envs.dart:DartHopper7LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper8Link-v1',
    entry_point='gym.envs.dart:DartHopper8LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper3Link2Foot-v1',
    entry_point='gym.envs.dart:DartHopper3Link2FootEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper4Link2Foot-v1',
    entry_point='gym.envs.dart:DartHopper4Link2FootEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper5Link2Foot-v1',
    entry_point='gym.envs.dart:DartHopper5Link2FootEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper4LinkSPD-v1',
    entry_point='gym.envs.dart:DartHopper4LinkSPDEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper5LinkSPD-v1',
    entry_point='gym.envs.dart:DartHopper5LinkSPDEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopper6Link2Foot-v1',
    entry_point='gym.envs.dart:DartHopper6Link2FootEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartReacher2Link-v1',
    entry_point='gym.envs.dart:DartReacher2LinkEnv',
    max_episode_steps=500,
)

register(
    id='DartReacher3Link-v1',
    entry_point='gym.envs.dart:DartReacher3LinkEnv',
    max_episode_steps=100,
)

register(
    id='DartReacher4Link-v1',
    entry_point='gym.envs.dart:DartReacher4LinkEnv',
    max_episode_steps=500,
)

register(
    id='DartReacher5Link-v1',
    entry_point='gym.envs.dart:DartReacher5LinkEnv',
    max_episode_steps=500,
)

register(
    id='DartReacher6Link-v1',
    entry_point='gym.envs.dart:DartReacher6LinkEnv',
    max_episode_steps=500,
)

register(
    id='DartReacher7Link-v1',
    entry_point='gym.envs.dart:DartReacher7LinkEnv',
    max_episode_steps=500,
)

register(
    id='DartHopperRSS-v1',
    entry_point='gym.envs.dart:DartHopperRSSEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopperAssist-v1',
    entry_point='gym.envs.dart:DartHopperAssistEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartSnake3Link-v1',
    entry_point='gym.envs.dart:DartSnake3LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartSnake4Link-v1',
    entry_point='gym.envs.dart:DartSnake4LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartSnake5Link-v1',
    entry_point='gym.envs.dart:DartSnake5LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartSnake6Link-v1',
    entry_point='gym.envs.dart:DartSnake6LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartSnake7Link-v1',
    entry_point='gym.envs.dart:DartSnake7LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartSnake8Link-v1',
    entry_point='gym.envs.dart:DartSnake8LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartSnake9Link-v1',
    entry_point='gym.envs.dart:DartSnake9LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartSnake10Link-v1',
    entry_point='gym.envs.dart:DartSnake10LinkEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

'''register(
    id='DartHopperRBF-v1',
    entry_point='gym.envs.dart:DartHopperRBFEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)'''

register(
    id='DartHopperBackpack-v1',
    entry_point='gym.envs.dart:DartHopperBackPackEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopperCont-v1',
    entry_point='gym.envs.dart:DartHopperEnvCont',
    reward_threshold=3800.0,
    max_episode_steps=1300,
)

register(
    id='DartCartPole-v1',
    entry_point='gym.envs.dart:DartCartPoleEnv',
    reward_threshold=950.0,
    max_episode_steps=1000,
)

register(
    id='DartDoubleInvertedPendulumEnv-v1',
    entry_point='gym.envs.dart:DartDoubleInvertedPendulumEnv',
    max_episode_steps=500,
)

register(
    id='DartReacher-v1',
    entry_point='gym.envs.dart:DartReacher2dEnv',
    reward_threshold=-3.75,
    max_episode_steps=250,
)

register(
    id='DartManipulator2d-v1',
    entry_point='gym.envs.dart:DartManipulator2dEnv',
    max_episode_steps=400,
)

register(
    id='DartReacher3d-v1',
    entry_point='gym.envs.dart:DartReacherEnv',
    reward_threshold=-200,
    max_episode_steps=500,
)

register(
    id='DartRobotWalk-v1',
    entry_point='gym.envs.dart:DartRobotWalk',
    max_episode_steps=1000,
)

register(
    id='DartDog-v1',
    entry_point='gym.envs.dart:DartDogEnv',
    max_episode_steps=300,
)

register(
    id='DartDogRobot-v1',
    entry_point='gym.envs.dart:DartDogRobotEnv',
    max_episode_steps=600,
)

register(
    id='DartHexapod-v1',
    entry_point='gym.envs.dart:DartHexapodEnv',
    max_episode_steps=500,
)

'''register(
    id='DartSwimmer-v1',
    entry_point='gym.envs.dart:DartSwimmerEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)'''

register(
    id='DartCartPole-v0',
    entry_point='gym.envs.dart:DartCartPoleImgEnv',
    reward_threshold=950.0,
    max_episode_steps=2000,
)

register(
    id='DartCartPoleSwingUp-v1',
    entry_point='gym.envs.dart:DartCartPoleSwingUpEnv',
    max_episode_steps=500,
)

register(
    id='DartPendulum-v1',
    entry_point='gym.envs.dart:DartPendulumEnv',
    max_episode_steps=500,
)

register(
    id='DartHalfCheetah-v1',
    entry_point='gym.envs.dart:DartHalfCheetahEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker2d-v1',
    entry_point='gym.envs.dart:DartWalker2dEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker2dBackpack-v1',
    entry_point='gym.envs.dart:DartWalker2dBackpackEnv',
    max_episode_steps=1000,
)

register(
    id='DartAnt-v1',
    entry_point='gym.envs.dart:DartAntEnv',
    max_episode_steps=500,
)

register(
    id='DartDarwin-v1',
    entry_point='gym.envs.dart:DartDarwinEnv',
    max_episode_steps=500,
)

register(
    id='DartDarwinSquat-v1',
    entry_point='gym.envs.dart:DartDarwinSquatEnv',
    max_episode_steps=500,
)

register(
    id='DartDarwinAltFoot-v1',
    entry_point='gym.envs.dart:DartDarwinNewFootEnv',
    max_episode_steps=1000,
)

register(
    id='DartFrankaTransport-v1',
    entry_point='gym.envs.dart:DartFrankaTransportEnv',
    max_episode_steps=1000,
)

register(
    id='DartOP3-v1',
    entry_point='gym.envs.dart:DartOP3Env',
    max_episode_steps=1000,
)

register(
    id='DartA1-v1',
    entry_point='gym.envs.dart:DartA1Env',
    max_episode_steps=1000,
)

register(
    id='DartWalker3d-v1',
    entry_point='gym.envs.dart:DartWalker3dEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker3dFull-v1',
    entry_point='gym.envs.dart:DartWalker3dFullEnv',
    max_episode_steps=300,
)

register(
    id='DartHumanWalker-v1',
    entry_point='gym.envs.dart:DartHumanWalkerEnv',
    max_episode_steps=500,
)

register(
    id='DartHumanBalance-v1',
    entry_point='gym.envs.dart:DartHumanBalanceEnv',
    max_episode_steps=500,
)

register(
    id='DartHumanoidBalance-v1',
    entry_point='gym.envs.dart:DartHumanoidBalanceEnv',
    max_episode_steps=500,
)

register(
    id='DartHumanBallWalker-v1',
    entry_point='gym.envs.dart:DartHumanBallWalkerEnv',
    max_episode_steps=500,
)

register(
    id='DartWalker3dRestricted-v1',
    entry_point='gym.envs.dart:DartWalker3dRestrictedEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker3dProject-v1',
    entry_point='gym.envs.dart:DartWalker3dProjectionEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker3dSPD-v1',
    entry_point='gym.envs.dart:DartWalker3dSPDEnv',
    max_episode_steps=900,
)

register(
    id='DartWalker2dPendulum-v1',
    entry_point='gym.envs.dart:DartWalker2dPendulumEnv',
    max_episode_steps=1000,
)

register(
    id='DartBallWalker-v1',
    entry_point='gym.envs.dart:DartBallWalkerEnv',
    max_episode_steps=1000,
)

register(
    id='DDPEnv-v1',
    entry_point='gym.envs.dart:DDPEnv',
    max_episode_steps=100,
)

'''register(
    id='DartSwing-v0',
    entry_point='gym.envs.dart:DartSwingEnv',
    reward_threshold=950.0,
    max_episode_steps=2000,
)'''

register(
    id='DartCrossPuzzle-v1',
    entry_point='gym.envs.dart:DartCrossPuzzle',
    max_episode_steps = 100,
)

register(
    id='DartBridgePuzzle-v1',
    entry_point='gym.envs.dart:DartBridgePuzzle',
    max_episode_steps = 650,
)

register(
    id='DartBallCircle-v1',
    entry_point='gym.envs.dart:DartBallCircle',
    max_episode_steps = 2000,
)

# Mujoco
# ----------------------------------------

# 2D
register(
    id='Reacher-v2',
    entry_point='gym.envs.mujoco:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Reacher3d-v1',
    entry_point='gym.envs.mujoco:Reacher3dEnv',
    max_episode_steps=500,
    reward_threshold=-200,
)

register(
    id='Pusher-v2',
    entry_point='gym.envs.mujoco:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Thrower-v2',
    entry_point='gym.envs.mujoco:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Striker-v2',
    entry_point='gym.envs.mujoco:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='InvertedPendulum-v2',
    entry_point='gym.envs.mujoco:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulum-v2',
    entry_point='gym.envs.mujoco:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id='HalfCheetah-v2',
    entry_point='gym.envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HalfCheetah-v3',
    entry_point='gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Hopper-v2',
    entry_point='gym.envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Hopper-v3',
    entry_point='gym.envs.hopper_v3:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='HopperRSS-v1',
    entry_point='gym.envs.mujoco:HopperRSSEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Swimmer-v2',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Swimmer-v3',
    entry_point='gym.envs.mujoco.swimmer_v3:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Walker2d-v2',
    max_episode_steps=1000,
    entry_point='gym.envs.mujoco:Walker2dEnv',
)

register(
    id='Walker2d-v3',
    max_episode_steps=1000,
    entry_point='gym.envs.mujoco.walker2d_v3:Walker2dEnv',
)

register(
    id='Ant-v2',
    entry_point='gym.envs.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Ant-v3',
    entry_point='gym.envs.mujoco.ant_v3:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v2',
    entry_point='gym.envs.mujoco:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='Humanoid-v3',
    entry_point='gym.envs.mujoco.humanoid_v3:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidStandup-v2',
    entry_point='gym.envs.mujoco:HumanoidStandupEnv',
    max_episode_steps=1000,
)

# Robotics
# ----------------------------------------

def _merge(a, b):
    a.update(b)
    return a

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    # Fetch
    register(
        id='FetchSlide{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:FetchSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndPlace{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchReach{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPush{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:FetchPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    # Hand
    register(
        id='HandReach{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='HandManipulateBlockRotateZ{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateZTouchSensors{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateZTouchSensors{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateParallel{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateParallelTouchSensors{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateParallelTouchSensors{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateXYZ{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateXYZTouchSensors{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateXYZTouchSensors{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockFull{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulateBlock{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockTouchSensors{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockTouchSensors{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggRotate{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandEggEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggRotateTouchSensors{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandEggTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggRotateTouchSensors{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:HandEggTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggFull{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulateEgg{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggTouchSensors{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandEggTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggTouchSensors{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:HandEggTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenRotate{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandPenEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenRotateTouchSensors{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandPenTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenRotateTouchSensors{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:HandPenTouchSensorsEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenFull{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulatePen{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenTouchSensors{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandPenTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'boolean'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenTouchSensors{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:HandPenTouchSensorsEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz', 'touch_get_obs': 'sensordata'}, kwargs),
        max_episode_steps=100,
    )

# Atari
# ----------------------------------------

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        register(
            id='{}-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
            max_episode_steps=10000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}Deterministic-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}Deterministic-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}NoFrameskip-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

        # No frameskip. (Atari has no entropy source, so these are
        # deterministic environments.)
        register(
            id='{}NoFrameskip-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )


# Unit test
# ---------

register(
    id='CubeCrash-v0',
    entry_point='gym.envs.unittest:CubeCrash',
    reward_threshold=0.9,
    )
register(
    id='CubeCrashSparse-v0',
    entry_point='gym.envs.unittest:CubeCrashSparse',
    reward_threshold=0.9,
    )
register(
    id='CubeCrashScreenBecomesBlack-v0',
    entry_point='gym.envs.unittest:CubeCrashScreenBecomesBlack',
    reward_threshold=0.9,
    )

register(
    id='MemorizeDigits-v0',
    entry_point='gym.envs.unittest:MemorizeDigits',
    reward_threshold=20,
    )
