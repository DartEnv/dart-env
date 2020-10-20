import numpy as np

def VAL2RADIAN(val):
    return (val - 2048) * 0.088 * np.pi / 180

def RADIAN2VAL(rad):
    return rad * 180 / np.pi / 0.088 + 2048

# Joint limits
JOINT_LOW_BOUND_VAL = np.array([
    0, 700, 1400, 0, 1600, 1000,
    1346, 1850,
    1800, 1400, 1800, 648, 1241, 1850,     1800, 2048, 1000, 2048, 1500, 1700
])

JOINT_UP_BOUND_VAL = np.array([
    4095, 2400, 3000, 4095, 3400, 2800,
    2632, 2600,
    2200, 2048, 3100, 2048, 2500, 2400,     2200, 2600, 2300, 3448, 2855, 2300
])

JOINT_LOW_BOUND = VAL2RADIAN(JOINT_LOW_BOUND_VAL)

JOINT_UP_BOUND = VAL2RADIAN(JOINT_UP_BOUND_VAL)

JOINT_LOW_BOUND_NEWFOOT_VAL = JOINT_LOW_BOUND_VAL[[0,1,2,3,4,5,6,7,8,9,10,11, 14,15,16,17]]

JOINT_UP_BOUND_NEWFOOT_VAL = JOINT_UP_BOUND_VAL[[0,1,2,3,4,5,6,7,8,9,10,11, 14,15,16,17]]

JOINT_LOW_BOUND_NEWFOOT = VAL2RADIAN(JOINT_LOW_BOUND_NEWFOOT_VAL)

JOINT_UP_BOUND_NEWFOOT = VAL2RADIAN(JOINT_UP_BOUND_NEWFOOT_VAL)


# Joint control limits
CONTROL_LOW_BOUND_VAL = np.array([
    1500, 1500, 1400, 1500, 1700, 1400,
    2000, 2000,
    # 1800, 1400, 1800, 648, 1241, 1850,     1800, 2048, 1000, 2048, 1500, 1700
    1950, 1950, 1800, 648, 1241, 1950,   1950, 2048, 1000, 2048, 1548, 1950
])

CONTROL_UP_BOUND_VAL = np.array([
    2500, 2300, 2600, 2500, 2600, 2600,
    2100, 2100,
    # 2200, 2048, 3100, 2048, 2500, 2400,     2200, 2600, 2300, 3448, 2855, 2300
    2150, 2048, 3100, 2048, 2548, 2150,   2150, 2150, 2300, 3448, 2855, 2150
])

CONTROL_LOW_BOUND = VAL2RADIAN(CONTROL_LOW_BOUND_VAL)

CONTROL_UP_BOUND = VAL2RADIAN(CONTROL_UP_BOUND_VAL)

CONTROL_LOW_BOUND_NEWFOOT_VAL = CONTROL_LOW_BOUND_VAL[[0,1,2,3,4,5,6,7,  8,9,10,11, 14,15,16,17]]

CONTROL_UP_BOUND_NEWFOOT_VAL = CONTROL_UP_BOUND_VAL[[0,1,2,3,4,5,6,7,  8,9,10,11, 14,15,16,17]]

CONTROL_LOW_BOUND_NEWFOOT = VAL2RADIAN(CONTROL_LOW_BOUND_NEWFOOT_VAL)

CONTROL_UP_BOUND_NEWFOOT = VAL2RADIAN(CONTROL_UP_BOUND_NEWFOOT_VAL)



############## predefined poses #######################
pose_squat_val = np.array([2509, 2297, 1714, 1508, 1816, 2376,
                                    2047, 2171,
                                    2032, 2039, 2795, 648, 1241, 2040,   2041, 2060, 1281, 3448, 2855, 2073])

pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                2048, 2048,
                                2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])

pose_left_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                2048, 2048,
                                 2032, 2039, 2795, 568, 1241, 2040, 2048, 2048, 2048, 2048, 2048, 2048])

pose_right_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                2048, 2048,
                                2048, 2048, 2048, 2048, 2048, 2048, 2041, 2060, 1281, 3525, 2855, 2073])

pose_squat_rad = VAL2RADIAN(pose_squat_val)
pose_stand_rad = VAL2RADIAN(pose_stand_val)
pose_left_stand_rad = VAL2RADIAN(pose_left_stand_val)
pose_right_stand_rad = VAL2RADIAN(pose_right_stand_val)

SIM2HW_JOINT_INDEX = [3,0,4,1,5,2,14,8,15,9,16,10,17,11,18,12,19,13,6,7]
HW2SIM_JOINT_INDEX = np.argsort(SIM2HW_JOINT_INDEX).astype(int).tolist()

def SIM2HW_INDEX(input):
    return (np.array(input)[SIM2HW_JOINT_INDEX])

def HW2SIM_INDEX(input):
    return (np.array(input)[HW2SIM_JOINT_INDEX])
######################################################














