import numpy as np

from src.actors.envs.environment import Environment
from src.verification.bounds.bounds import HyperRectangleBounds
from operator import __ge__, __eq__


class VcasConstants:
    # Sea-level gravitational acceleration constant, in ft / s / s.
    G = 32.2

    # New advisories are given once every DELTA_T second(s).
    DELTA_T = 1

    # Advisories.
    (COC, DNC, DND, DES1500, CL1500, SDES1500, SCL1500, SDES2500, SCL2500) = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    # Number of advisories.
    N_ADVISORIES = 9

    # Scenario state.
    (INTRUDER_ALTITUDE, OWNSHIP_VERTICAL_CLIMBRATE, TAU, ACCELERATION, ADVISORY) = (0, 1, 2, 3, 4)

    # Dimension of state.
    N_STATE_VARS = 3

    # Bounds of scenario state varables.
    MIN_INTRUDER_ALTITUDE, MAX_INTRUDER_ALTITUDE = -3000, 3000  # ft
    MIN_OWNSHIP_VERTICAL_CLIMBRATE, MAX_OWNSHIP_VERTICAL_CLIMBRATE = -2500, 2500  # ft / s
    MIN_TAU, MAX_TAU = 0, 40  # seconds
    MIN_ACCELERATION, MAX_ACCELERATION = -G / 3, G / 3  # ft / s / s

    # Map of next possible (quantised) acceleration values given an advisory.
    NEXT_ACCELERATION = {
        COC: [-G / 8, 0, G / 8],
        DNC: [-G / 3, -G * 7 / 24, -G / 4],
        DND: [G / 4, G * 7 / 24, G / 3],
        DES1500: [-G / 3, -G * 7 / 24, -G / 4],
        CL1500: [G / 4, G * 7 / 24, G / 3],
        SDES1500: [-G / 3, -G / 3, -G / 3],
        SCL1500: [G / 3, G / 3, G / 3],
        SDES2500: [-G / 3, -G / 3, -G / 3],
        SCL2500: [G / 3, G / 3, G / 3],
    }

    # Map of next possible (quantised) acceleration values given an advisory.
    NEXT_ACCELERATION_BF = {
        COC: 3,
        DNC: 3,
        DND: 3,
        DES1500: 3,
        CL1500: 3,
        SDES1500: 1,
        SCL1500: 1,
        SDES2500: 1,
        SCL2500: 1,
    }

    # Branching factor of environment.
    BF = 3

    INPUT_MEAN_VALUES = np.array([0.0, 0.0, 20.0])
    INPUT_RANGES = np.array([16000.0, 5000.0, 40.0])
    OUTPUT_MEAN = -0.7194709316423972
    OUTPUT_RANGE = 26.24923585890485


def normalise_input(values):
    return (values - VcasConstants.INPUT_MEAN_VALUES) / VcasConstants.INPUT_RANGES


def normalise(value, mean, range):
    return (value - mean) / range


def get_deterministic_non_deterministic_case_constraints(constrs_manager, delta):
    """

    :param constrs_manager:
    :param delta: is a binary variable for an advisory adv that is true
        when the previous advisory is exactly adv
    :return:
      nondet: binary variable for the non-deterministic case
      det: binary variable for the deterministic case
      constrs: constraints
    """

    # Binary variable for checking whether an acceleration should be chosen non-deterministically.
    [nondet] = constrs_manager.create_binary_variables(1)
    # Binary variable for checking whether an acceleration should not be chosen.
    [det] = constrs_manager.create_binary_variables(1)

    # We can only be in either the deterministic or non-deterministic case.
    ## delta[adv] == 1 >> nondet + det == 1
    ## det == 1 >> delta[adv] == 1
    ## nondet == 1 >> delta[adv] == 1
    #
    # Same logic without indicator constraints
    constrs = [
        constrs_manager.get_linear_constraint(
            [nondet, det, delta], [1, 1, -1], 0, sense=__ge__),
        constrs_manager.get_linear_constraint(
            [delta, det], [1, -1], 0, sense=__ge__),
        constrs_manager.get_linear_constraint(
            [delta, nondet], [1, -1], 0, sense=__ge__)
    ]

    return nondet, det, constrs


def get_advisories_delta_bounds(adv_bounds):
    adv_bounds_lower, adv_bounds_upper = adv_bounds

    delta_lower_bounds = [0 for _ in range(VcasConstants.N_ADVISORIES)]
    delta_upper_bounds = [1 for _ in range(VcasConstants.N_ADVISORIES)]

    if adv_bounds_lower == adv_bounds_upper:
        delta_upper_bounds = [0 for _ in range(VcasConstants.N_ADVISORIES)]
        delta_lower_bounds[int(adv_bounds_lower)] = 1
        delta_upper_bounds[int(adv_bounds_lower)] = 1
    else:
        for i in range(VcasConstants.N_ADVISORIES):
            if adv_bounds_lower > i or adv_bounds_upper < i:
                delta_upper_bounds[i] = 0

    return delta_lower_bounds, delta_upper_bounds


def get_compliance_constraint(constrs_manager, advisory, prev_vertical_climbrate):
    """
    Return a linear constraint representing the conditions for compliance for a given advisory.
    :param advisory: The advisory issued by the agent in range(1, 9).
    :param prev_vertical_climbrate: A variable representing ownship's vertical climbrate
    at the previous timestep.
    :return:
    """
    compliance_map = {
        VcasConstants.DNC: constrs_manager.get_le_constraint(prev_vertical_climbrate, 0),
        VcasConstants.DND: constrs_manager.get_ge_constraint(prev_vertical_climbrate, 0),
        VcasConstants.DES1500: constrs_manager.get_le_constraint(prev_vertical_climbrate, -1500),
        VcasConstants.CL1500: constrs_manager.get_ge_constraint(prev_vertical_climbrate, 1500),
        VcasConstants.SDES1500: constrs_manager.get_le_constraint(prev_vertical_climbrate, -1500),
        VcasConstants.SCL1500: constrs_manager.get_ge_constraint(prev_vertical_climbrate, 1500),
        VcasConstants.SDES2500: constrs_manager.get_le_constraint(prev_vertical_climbrate, -2500),
        VcasConstants.SCL2500: constrs_manager.get_ge_constraint(prev_vertical_climbrate, 2500),
    }
    return compliance_map[advisory]


class VcasEnv(Environment):

    def __init__(self):
        """
        VcasEnv represents the pilot of the VerticalCAS scenario. It is responsible for selecting
        an acceleration partially determined by the previous advisory (given by the agent), and
        transitioning the system into the the next state.
        :param state_space_dim: The dimension of the *environment* state space.
        """
        super(VcasEnv, self).__init__(3)

        # Constants.
        # Dimension of the state space; this is len([position, vertical_climbrate, tau, acceleration, prev_adv]) = 5.
        self.STATE_SPACE_DIM = 5

    def get_branching_factor_opt(self, input_state_vars, action_vars):
        #TODO implement
        [advisory_var] = action_vars
        l, u = advisory_var.lb, advisory_var.ub

        bf = np.max([VcasConstants.NEXT_ACCELERATION_BF[adv] for adv in range(l, u+1)])

        return bf

    def get_constraints_for_transition(self, i, constrs_manager, action_vars, input_state_vars):
        """
        Add the constraints for the transition function itself, create constraints to determine the
        Gurobi variables representing the next state to be input back into the agent. Constraints
        are only added to the Gurobi model by the caller, to reduce side-effects in this function.
        :param i: The transition function used to move to the next state.
        :param constrs_manager: Gurobi constraints representing the current state/situation.
        :param action_vars: Gurobi variables representing the agent's action.
        :param input_state_vars: Gurobi variables representing the state of the **scenario**.
        :side-effects: Modifies constrs_manager when adding variables.
        :return: The Gurobi variables representing the state after the transition has been made.
        """

        # A list used to contain Gurobi constraints to be added to the constraints manager.
        constrs_to_add = []

        # The agent's action represents the advisory.
        advisory_var = action_vars[0]
        adv_bounds = constrs_manager.get_variable_bounds([advisory_var])

        ###################################################################
        # We need binary variables for each of the advisories produced by agent since the next
        # acceleration is dependent on the advisory. We must therefore encode a disjunction to
        # handle each case.
        delta_lower_bounds, delta_upper_bounds = get_advisories_delta_bounds(adv_bounds.get_dimension_bounds(0))
        delta_adv = constrs_manager.create_binary_variables(VcasConstants.N_ADVISORIES,
                                                            lbs=delta_lower_bounds, ubs=delta_upper_bounds)
        constrs_to_add.append(
            constrs_manager.get_sum_constraint(delta_adv, 1)
        )

        constrs_manager.update()

        ##################################################################
        # Acceleration is chosen by the environment depending on i.
        # Compute the bounds of the acceleration.
        acc_var_lower = VcasConstants.MAX_ACCELERATION
        acc_var_upper = VcasConstants.MIN_ACCELERATION
        for adv in range(VcasConstants.N_ADVISORIES):
            if delta_adv[adv].ub == 0:
                continue

            next_acceleration = VcasConstants.NEXT_ACCELERATION[adv][i]

            acc_var_lower = min(acc_var_lower, next_acceleration)
            acc_var_upper = max(acc_var_upper, next_acceleration)

        # Create the acceleration variable
        [acc_var] = constrs_manager.create_state_variables(1, lbs=[acc_var_lower], ubs=[acc_var_upper])
        constrs_manager.add_variable_bounds([acc_var], HyperRectangleBounds([acc_var_lower], [acc_var_upper]))

        # Add indicator constraints for producing the next acceleration for all advisories.
        for adv in range(VcasConstants.N_ADVISORIES):
            if delta_adv[adv].ub == 0:
                continue

            # The next acceleration corresponding to idx
            next_acceleration = VcasConstants.NEXT_ACCELERATION[adv][i]

            # Link advisory issued by agent to current advisory in this loop.
            # delta_adv[adv] == 1 >> advisory_var == adv
            constrs_to_add.append(
                constrs_manager.create_indicator_constraint(
                    delta_adv[adv], 1,
                    constrs_manager.get_assignment_constraint(advisory_var, adv)))

            # Acceleration should simply be next_acceleration in the case of COC
            if adv == VcasConstants.COC:
                # delta_adv[COC] == 1 >> next_state[acceleration] = next_acceleration
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(
                        delta_adv[adv], 1,
                        constrs_manager.get_assignment_constraint(acc_var, next_acceleration)
                    ))

            # For other advisories, we also need to check
            # whether the previous advisory and the issued advisories are the same and
            # whether the current climbrate is compliant with the advisory.
            # When it is, we are in the deterministic case and the acceleration should be 0,
            # when not, we are in the non-deterministic case and we choose acceleration according to idx
            else:
                # nondet and det are binary variable for the non-deterministic and deterministic cases, resp.
                nondet, det, det_non_det_constrs = \
                    get_deterministic_non_deterministic_case_constraints(constrs_manager, delta_adv[adv])
                constrs_to_add.extend(det_non_det_constrs)

                constrs_manager.update()

                # Deterministic case only occurs when advisory is the same as the previous one
                # and if the pilot is compliant with the issued advisory, otherwise, we're in
                # the non-deterministic case.

                ## det == 1 >> advisory == input_state_vars[VcasConstants.ADVISORY]
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(
                        det, 1,
                        constrs_manager.get_equality_constraint(advisory_var, input_state_vars[VcasConstants.ADVISORY])))

                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(
                        det, 1,
                        get_compliance_constraint(constrs_manager, adv,
                                                  input_state_vars[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE]
                                                  )))

                # In the deterministic case, maintain the same climbrate (hence, acceleration should be 0).
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(
                        det, 1, constrs_manager.get_assignment_constraint(acc_var, 0)))

                # In the non-deterministic case, set the acceleration to be next_acceleration
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(
                        nondet, 1, constrs_manager.get_assignment_constraint(acc_var, next_acceleration)))

        #################################################################
        # Creating and computing the next state variables and constraints.
        input_variable_bounds = constrs_manager.get_variable_bounds(input_state_vars)
        acc_bounds = constrs_manager.get_variable_bounds([acc_var])

        # Explicitly compute the lower and upper bounds of the next state variables
        # (for the first 3 components).
        next_state_lower, next_state_upper = self.compute_next_state_bounds(input_variable_bounds, acc_bounds)
        # Append the bounds of the acceleration and the advisory as the last 2 components are exactly those.
        next_state_lower.extend([acc_bounds.get_lower()[0], adv_bounds.get_lower()[0]])
        next_state_upper.extend([acc_bounds.get_upper()[0], adv_bounds.get_upper()[0]])

        # Create next state variables with the bounds provided.
        next_state_vars = \
            constrs_manager.create_state_variables(len(input_state_vars), lbs=next_state_lower, ubs=next_state_upper)
        constrs_manager.add_variable_bounds(next_state_vars, HyperRectangleBounds(next_state_lower, next_state_upper))

        # Compute and add constraints for the first 3 components of the next state.
        constrs_to_add.extend(
            self.compute_constraints_for_next_state(constrs_manager, next_state_vars, input_state_vars, advisory_var))
        # Add the constraints setting the last 2 components of the next state to the accelerations and advisories
        constrs_to_add.extend([
            constrs_manager.get_equality_constraint(next_state_vars[VcasConstants.ACCELERATION], acc_var),
            constrs_manager.get_equality_constraint(next_state_vars[VcasConstants.ADVISORY], advisory_var)
        ])

        # Return the variables representing the next state.
        return next_state_vars, constrs_to_add

    @staticmethod
    def compute_constraints_for_next_state(constrs_manager, next_state_vars, input_state_vars, advisory):
        return [
            # altitude = prev_altitude - prev_climbrate * dt - 0.5 * acceleration * dt * dt
            constrs_manager.get_linear_constraint(
                [next_state_vars[VcasConstants.INTRUDER_ALTITUDE],
                input_state_vars[VcasConstants.INTRUDER_ALTITUDE],
                input_state_vars[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE],
                next_state_vars[VcasConstants.ACCELERATION]],
                [1, -1, VcasConstants.DELTA_T, 0.5*VcasConstants.DELTA_T**2],
                0),
            # climbrate = prev_climbrate + acceleration * dt
            constrs_manager.get_linear_constraint(
                [next_state_vars[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE],
                 input_state_vars[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE],
                 next_state_vars[VcasConstants.ACCELERATION]],
                [1, -1, -VcasConstants.DELTA_T],
                0),
            # tau = prev_tau - dt
            constrs_manager.get_linear_constraint(
                [next_state_vars[VcasConstants.TAU], input_state_vars[VcasConstants.TAU]],
                [1, -1],
                -VcasConstants.DELTA_T)
            ]

    @staticmethod
    def compute_next_state_bounds(input_variable_bounds, acc_bounds):
        input_variable_bounds_lower = input_variable_bounds.get_lower()
        input_variable_bounds_upper = input_variable_bounds.get_upper()
        acc_lower, acc_upper = acc_bounds.get_dimension_bounds(0)

        next_state_lower = \
            [input_variable_bounds_lower[VcasConstants.INTRUDER_ALTITUDE] -
             VcasConstants.DELTA_T * input_variable_bounds_upper[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE] -
             VcasConstants.DELTA_T**2 * 0.5 * acc_upper,
             input_variable_bounds_lower[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE] + VcasConstants.DELTA_T * acc_lower,
             input_variable_bounds_lower[VcasConstants.TAU] - VcasConstants.DELTA_T,
            ]

        next_state_upper = \
            [input_variable_bounds_upper[VcasConstants.INTRUDER_ALTITUDE] -
             VcasConstants.DELTA_T * input_variable_bounds_lower[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE] -
             VcasConstants.DELTA_T**2 * 0.5 * acc_lower,
             input_variable_bounds_upper[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE] + acc_upper,
             input_variable_bounds_upper[VcasConstants.TAU] - VcasConstants.DELTA_T
            ]
        return next_state_lower, next_state_upper
