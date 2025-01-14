import collections
from operator import __le__, __ge__

import numpy as np


class VcasConstants:
    # Sea-level gravitational acceleration constant, in ft / s / s.
    G = 32.2

    # Number of advisories.
    N_ADVISORIES = 9

    # Advisories.
    (COC, DNC, DND, DES1500, CL1500, SDES1500, SCL1500, SDES2500, SCL2500) = range(N_ADVISORIES)

    SCENARIO_STATE_DIM = 8

    # Scenario state, used to allocate the entire "state" of Gurobi variables representing the scenario.
    (INT_ALT_IDX, OWN_CLIMBRATE_IDX, INT_CLIMBRATE_IDX, TAU_IDX, OWN_ACC_IDX, INT_ACC_IDX, OWN_ADV_IDX, INT_ADV_IDX) \
        = range(SCENARIO_STATE_DIM)

    # Dimension of (actual) state.
    N_STATE_VARS = 4

    # Bounds of scenario state varables.
    MIN_INT_ALT, MAX_INT_ALT = -8000, 8000  # ft
    MIN_OWN_CLIMBRATE, MAX_OWN_CLIMBRATE = -100, 100  # ft / s
    MIN_INT_CLIMBRATE, MAX_INT_CLIMBRATE = -100, 100  # ft / s
    MIN_TAU, MAX_TAU = 0, 40  # seconds
    MIN_OWN_ACC, MAX_OWN_ACC = -11.7, 11.7  # ft / s / s
    MIN_INT_ACC, MAX_INT_ACC = -11.7, 11.7  # ft / s / s
    # MIN_OWN_ACC_INT, MAX_OWN_ACC_INT = -11, 11  # ft / s / s
    # MIN_INT_ACC_INT, MAX_INT_ACC_INT = -11, 11  # ft / s / s

    # ADV_INFO :: Adv -> (Accels, Sense, CompliantClimbrate)
    AdvInfo = collections.namedtuple('AdvInfo', 'accelerations, sense, climbrate')
    # ADV_INFO = {COC:      AdvInfo([0.0, 3.0, -3.0], None, None),
    #             DNC:      AdvInfo([-8.33, -9.33, -7.33], __le__, 0),
    #             DND:      AdvInfo([8.33, 9.33, 7.33], __ge__, 0),
    #             DES1500:  AdvInfo([-8.33, -9.33, -7.33], __le__, -1500),
    #             CL1500:   AdvInfo([8.33, 9.33, 7.33], __ge__, 1500),
    #             SDES1500: AdvInfo([-10.7, -11.7, -9.7], __le__, -1500),
    #             SCL1500:  AdvInfo([10.7, 11.7, 9.7], __ge__, 1500),
    #             SDES2500: AdvInfo([-10.7, -11.7, -9.7], __le__, -2500),
    #             SCL2500:  AdvInfo([10.7, 11.7, 9.7], __ge__, 2500)}

    # BF 2
    ADV_INFO = {COC:      AdvInfo([-3.0, 3.0], None, None),
                DNC:      AdvInfo([-9.33, -7.33], __le__, 0),
                DND:      AdvInfo([7.33, 9.33], __ge__, 0),
                DES1500:  AdvInfo([-9.33, -7.33], __le__, -1500),
                CL1500:   AdvInfo([7.33, 9.33], __ge__, 1500),
                SDES1500: AdvInfo([-11.7, -9.7], __le__, -1500),
                SCL1500:  AdvInfo([9.7, 11.7], __ge__, 1500),
                SDES2500: AdvInfo([-11.7, -9.7], __le__, -2500),
                SCL2500:  AdvInfo([9.7, 11.7], __ge__, 2500)}

    # ADV_INFO_INT = {COC: AdvInfo([0, 3, -3], None, None),
    #                 DNC: AdvInfo([-8, -9, -7], __le__, 0),
    #                 DND: AdvInfo([8, 9, 7], __ge__, 0),
    #                 DES1500: AdvInfo([-8, -9, -7], __le__, -1500),
    #                 CL1500: AdvInfo([8, 9, 7], __ge__, 1500),
    #                 SDES1500: AdvInfo([-10, -11, -9], __le__, -1500),
    #                 SCL1500: AdvInfo([10, 11, 9], __ge__, 1500),
    #                 SDES2500: AdvInfo([-10, -11, -9], __le__, -2500),
    #                 SCL2500: AdvInfo([10, 11, 9], __ge__, 2500)}

    # Branching factor of environment.
    BF = 2

    INPUT_MEAN_VALUES = np.array([0.0, 0.0, 0.0, 20.0])
    INPUT_RANGES = np.array([16000.0, 200.0, 200.0, 40.0])
    OUTPUT_MEAN = -0.432599379632
    OUTPUT_RANGE = 3.102300001