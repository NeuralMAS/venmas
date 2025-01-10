import itertools
import numpy as np

from src.actors.envs.multi_agent_environment import AbstractMultiAgentEnvironment
from resources.vcas.two_agent_vcasenv_constants import VcasConstants
from src.utils.constraint import LinearConstraint, WeightedSum
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.complete.constrmanager.custom_constraints_manager import CustomConstraintsManager
from operator import __ge__, __eq__


def normalise_input(values):
    return (values - VcasConstants.INPUT_MEAN_VALUES) / VcasConstants.INPUT_RANGES


def normalise(value, mean, range):
    return (value - mean) / range


def get_intruder_altitude(input_state_vars, acceleration_own, acceleration_int):
    """
    Compute the altitude of the intruder at the next timestep.
    :param acceleration_own: Gurobi variable representing the last acceleration chosen for ownship.
    :param acceleration_int: Gurobi variable representing the last acceleration chosen for intruder.
    :param input_state_vars: List of Gurobi variables representing the current state.
    :return: A Gurobi variable representing the next intruder altitude.
    """
    prev_altitude = input_state_vars[VcasConstants.INT_ALT_IDX]
    prev_climbrate_own = input_state_vars[VcasConstants.OWN_CLIMBRATE_IDX]
    prev_climbrate_int = input_state_vars[VcasConstants.INT_CLIMBRATE_IDX]
    new_alt = prev_altitude - prev_climbrate_own - 0.5 * acceleration_own + prev_climbrate_int + 0.5 * acceleration_int
    return new_alt


def get_ownship_vertical_climbrate(input_state_vars, own_acceleration):
    """
    Compute the vertical climbrate of the ownship at the next timestep.
    :param own_acceleration: Gurobi variable representing the last ownship acceleration chosen.
    :param input_state_vars: List of Gurobi variables representing the current state.
    :return: A Gurobi variable representing the next vertical climbrate.
    """
    prev_climbrate = input_state_vars[VcasConstants.OWN_CLIMBRATE_IDX]
    return prev_climbrate + own_acceleration


def get_intruder_vertical_climbrate(input_state_vars, int_acceleration):
    """
    Compute the vertical climbrate of the ownship at the next timestep.
    :param int_acceleration: Gurobi variable representing the last intruder acceleration chosen.
    :param input_state_vars: List of Gurobi variables representing the current state.
    :return: A Gurobi variable representing the next vertical climbrate.
    """
    prev_climbrate = input_state_vars[VcasConstants.INT_CLIMBRATE_IDX]
    return prev_climbrate + int_acceleration


def get_tau(input_state_vars):
    """
    Compute the time to loss of horizontal separation of the two aircraft at the next timestep.
    :param input_state_vars: List of Gurobi variables representing the current state.
    :return: A Gurobi variable representing the next value of tau.
    """
    tau = input_state_vars[VcasConstants.TAU_IDX]
    return tau - 1


class TwoAgentOwnshipVcasEnv(AbstractMultiAgentEnvironment):

    def __init__(self, state_space_dim):
        """
        VcasEnv represents the pilot of the VerticalCAS scenario. It is responsible for selecting
        an acceleration partially determined by the previous advisory (given by the agent), and
        transitioning the system into the next state.
        :param state_space_dim: The dimension of the *environment* state space.
        """
        super(TwoAgentOwnshipVcasEnv, self).__init__(VcasConstants.BF)

        # Constants.
        self.STATE_SPACE_DIM = state_space_dim
        self.NUM_AGENTS_GAMMA = 1  # Gamma = { ownship }
        self.NUM_AGENTS_NOT_GAMMA = 1  # not(Gamma) = { intruder }

    def get_constraints_for_transition(self, constraints_manager, joint_action_vars_gamma, joint_action_vars_not_gamma, state_vars):

        # A list used to contain Gurobi constraints to be added to the constraints manager.
        constrs_to_add = []

        # Set of (binary variable, binary value) tuples. Used when nesting indicator constraints in
        # MILP visitor.
        binvars = set()

        # Compute bounds for the next state.
        input_variable_bounds = constraints_manager.get_variable_bounds(state_vars)
        input_variable_bounds_lower = input_variable_bounds.get_lower()
        input_variable_bounds_upper = input_variable_bounds.get_upper()

        # Get each agent's action, assuming Agt = {own}, and not(Agt) = {int}.
        own_see = joint_action_vars_gamma[0]
        int_see = joint_action_vars_not_gamma[0]
        # int_see = joint_see_vars_agt[1] if not joint_see_vars_not_agt[0] else joint_see_vars_not_agt[0]

        own_see_bounds = constraints_manager.get_variable_bounds(own_see)
        int_see_bounds = constraints_manager.get_variable_bounds(int_see)

        # For clarity, explicitly compute the greatest possible lower bound of the next state
        # variables. This will become the next lower bound of the state variables for the
        # next timestep.
        # Note: Bounds for acceleration and advisory remain the loosest possible, since it is not
        # possible to know their values in advance by using information from the previous timestep,
        # as their values depend on the output of one of the agent's neural networks.
        next_state_lower = \
            self.compute_next_state_lower_bounds(
                input_variable_bounds_lower, input_variable_bounds_upper, own_see_bounds, int_see_bounds)

        # Explicitly compute the lowest possible upper bounds of the next state variables.
        # This will become the next upper bounds for the state variables of the next timestep.
        # Bounds for acceleration and advisory remain the loosest possible (see above comment).
        next_state_upper = \
            self.compute_next_state_upper_bounds(
                input_variable_bounds_lower, input_variable_bounds_upper, own_see_bounds, int_see_bounds)

        # Create next state variables with the bounds provided.
        grb_vars_next_state = \
            constraints_manager.create_state_variables(len(state_vars), lbs=next_state_lower, ubs=next_state_upper)
        constraints_manager.add_variable_bounds(grb_vars_next_state, HyperRectangleBounds(next_state_lower, next_state_upper))

        # First component of each agent's action represents its advisory.
        own_advisory = own_see[0]
        int_advisory = int_see[0]

        # Compute and add constraints representing the next state of the scenario.
        constrs_to_add.extend(
            self.compute_constraints_for_next_state(
                own_advisory, int_advisory, grb_vars_next_state, state_vars, constraints_manager))

        # *** DETERMINISTIC TRANSITION CODE FINISHES HERE ***

        # Return the Gurobi variables representing the next state.
        return grb_vars_next_state, constrs_to_add

    def get_constraints_for_joint_protocol(self, constrs_manager, input_state_vars,
                                                 joint_action_gamma_idx, joint_action_not_gamma_idx,
                                                 gamma_see_vars, not_gamma_see_vars,
                                                 grb_vars_next_state,
                                                 binvars):
        """
        Get the constraints for a global joint action from agents in Agt.
        Side-effects: modifies constrs_mgr.
        :param constrs_manager: Gurobi constraints manager
        :param input_state_vars:  Gurobi variables representing current state.
        :param joint_action_gamma_idx: Index into set of joint actions of agents in Gamma.
        :param joint_action_not_gamma_idx: Index into set of joint actions of agents in not(Gamma).
        :param gamma_see_vars: Gurobi variables representing the (joint) percept of agents in Gamma.
        :param not_gamma_see_vars: Gurobi variables representing the (joint) percept of agents in not(Gamma).
        :param grb_vars_next_state: Gurobi variables used to encode the next state.
        :param binvars:
        :return:
        """

        def get_compliance_constraint(advisory, prev_vertical_climbrate):
            """
            Return a linear constraint representing the conditions for compliance for a given advisory.
            :param advisory: The advisory issued by the agent in range(1, 9).
            :param prev_vertical_climbrate: A variable representing ownship's vertical climbrate
            at the previous timestep.
            :param constrs_manager: Gurobi constraints representing the current state/situation.
            :return:
            """
            info = VcasConstants.ADV_INFO[advisory]
            return constrs_manager.get_inequality_constraint(prev_vertical_climbrate, info.climbrate, info.sense)

        # Set of (binary variable, binary value) tuples. Used when nesting indicator constraints in
        # MILP visitor.
        binvars = set()

        # Generate list of indices into joint protocol for gamma
        joint_protocol_gamma_indices = list(itertools.product(
            range(self.branching_factor), repeat=len(gamma_see_vars)))

        # Generate list of indices into joint protocol for not gamma
        joint_protocol_not_gamma_indices = list(itertools.product(
            range(self.branching_factor), repeat=len(not_gamma_see_vars)))

        # Here assuming Gamma = { own } and not(Gamma) = { int }.
        own_advisory_var = gamma_see_vars[0][0]
        int_advisory_var = not_gamma_see_vars[0][0]

        own_advisory_var_bounds = constrs_manager.get_variable_bounds([own_advisory_var])
        int_advisory_var_bounds = constrs_manager.get_variable_bounds([int_advisory_var])

        # int_advisory_var = gamma_see_vars[1][0] if not not_gamma_see_vars[0] else not_gamma_see_vars[0][0]

        own_climbrate_idx = VcasConstants.OWN_CLIMBRATE_IDX
        own_acc_idx = VcasConstants.OWN_ACC_IDX
        own_adv_idx = VcasConstants.OWN_ADV_IDX

        int_climbrate_idx = VcasConstants.INT_CLIMBRATE_IDX
        int_acc_idx = VcasConstants.INT_ACC_IDX
        int_adv_idx = VcasConstants.INT_ADV_IDX

        # agt_indices = self.get_indices_gamma(joint_action_gamma_idx)
        # not_agt_indices = self.get_indices_not_gamma(not_gamma_joint_action_idx)

        # We need binary variables for each of the advisories produced by ownship agent since the next
        # acceleration is dependent on the advisory. We must therefore encode a disjunction to
        # handle each case.
        constrs_to_add = []

        next_accelerations_own = VcasConstants.ADV_INFO[VcasConstants.COC].accelerations
        #
        # if own_advisory_var_bounds.get_lower() == own_advisory_var_bounds.get_upper() \
        #         and not isinstance(constrs_manager, CustomConstraintsManagerWithBounds):  # We know the exact advisory
        #     self.optimise_adv_own(binvars, constrs_manager, constrs_to_add, get_compliance_constraint,
        #                           grb_vars_next_state, input_state_vars, joint_action_gamma_idx,
        #                           joint_protocol_gamma_indices, next_accelerations_own, own_acc_idx,
        #                           own_adv_idx, own_advisory_var, own_advisory_var_bounds,
        #                           own_climbrate_idx)
        # else:

        delta_own_adv = constrs_manager.create_binary_variables(VcasConstants.N_ADVISORIES)
        constrs_to_add.append(
            constrs_manager.get_sum_constraint(delta_own_adv, 1)
        )
        constrs_manager.update()

        # Add constraints immediately in the case of COC, since we do not require a nested
        # conditional to compute the next acceleration.
        # delta_own[COC] == 1 >> own_chosen_adv == COC
        constrs_to_add.append(
            constrs_manager
                .create_indicator_constraint(delta_own_adv[VcasConstants.COC],
                                             1,
                                             constrs_manager.get_assignment_constraint(
                                                 own_advisory_var,
                                                 VcasConstants.COC)
                                             )
        )

        # delta_own[COC] == 1 >> next_state[ownship_acceleration] = next_accelerations[COC][i]
        constrs_to_add.append(
            constrs_manager
                .create_indicator_constraint(delta_own_adv[VcasConstants.COC],
                                             1,
                                             constrs_manager.get_assignment_constraint(
                                                            grb_vars_next_state[VcasConstants.OWN_ACC_IDX],
                                                            next_accelerations_own[joint_protocol_gamma_indices[joint_action_gamma_idx][0]])))

        # # Add indicator constraints for producing the next acceleration for all the remaining advisories.
        for adv in range(1, VcasConstants.N_ADVISORIES):
            # Link advisory issued by agent to current advisory in this loop.
            # delta_own_adv[adv] == 1 >> own_advisory_var == adv
            constrs_to_add.append(
                constrs_manager
                    .create_indicator_constraint(delta_own_adv[adv], 1,
                                                 constrs_manager.get_assignment_constraint(own_advisory_var, adv)))

            # Binary variable for checking whether an acceleration should be chosen
            # non-deterministically.
            [nondet] = constrs_manager.create_binary_variables(1)

            # Binary variable for checking whether an acceleration should not be chosen.
            [det] = constrs_manager.create_binary_variables(1)
            constrs_manager.update()

            # We can only be in either the deterministic or non-deterministic case.
            ## delta[adv] == 1 >> nondet + det == 1
            ## det == 1 >> delta[adv] == 1
            ## nondet == 1 >> delta[adv] == 1
            # Same logic without indicator constraints
            if isinstance(constrs_manager, CustomConstraintsManager):
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([nondet, det, delta_own_adv[adv]], [1, 1, -1]), __ge__, 0))
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([delta_own_adv[adv], det], [1, -1]), __ge__, 0))
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([delta_own_adv[adv], nondet], [1, -1]), __ge__, 0))
            else:
                constrs_to_add.append(nondet + det - delta_own_adv[adv] >= 0)
                constrs_to_add.append(delta_own_adv[adv] - det >= 0)
                constrs_to_add.append(delta_own_adv[adv] - nondet >= 0)

            self.encode_det_nondet(adv, binvars, constrs_manager, constrs_to_add, det,
                                   get_compliance_constraint, grb_vars_next_state, input_state_vars,
                                   joint_action_gamma_idx, joint_protocol_gamma_indices, nondet, own_acc_idx,
                                   own_adv_idx, own_advisory_var, own_climbrate_idx, 0)

        # **** Do the same for intruder ****
        next_accelerations_int = VcasConstants.ADV_INFO[VcasConstants.COC].accelerations
        # if int_advisory_var_bounds.get_lower() == int_advisory_var_bounds.get_upper() \
        #         and not isinstance(constrs_manager, CustomConstraintsManagerWithBounds):  # We know the exact advisory
        #     self.optimise_adv_intruder(binvars, constrs_manager, constrs_to_add, get_compliance_constraint,
        #                                grb_vars_next_state, input_state_vars, int_acc_idx, int_adv_idx,
        #                                int_advisory_var, int_advisory_var_bounds, int_climbrate_idx,
        #                                joint_action_gamma_idx, joint_action_not_gamma_idx, next_accelerations_int)
        # else:

        # Only possible alternative is having matrix of binary variables 9x9. Very expensive.
        delta_int_adv = constrs_manager.create_binary_variables(VcasConstants.N_ADVISORIES)
        constrs_to_add.append(
            constrs_manager.get_sum_constraint(delta_int_adv, 1)
        )
        constrs_manager.update()
        # Add constraints immediately in the case of COC, since we do not require a nested
        # conditional to compute the next acceleration.
        constrs_to_add.append(
            constrs_manager
                .create_indicator_constraint(
                delta_int_adv[VcasConstants.COC], 1,
                constrs_manager.get_assignment_constraint(int_advisory_var, VcasConstants.COC))
        )

        constrs_to_add.append(
            constrs_manager
                .create_indicator_constraint(
                delta_int_adv[VcasConstants.COC], 1,
                constrs_manager.get_assignment_constraint(
                    grb_vars_next_state[int_acc_idx],
                    next_accelerations_int[joint_protocol_not_gamma_indices[joint_action_not_gamma_idx][0]])))

        # # Add indicator constraints for producing the next acceleration for all the remaining advisories.
        for adv in range(1, VcasConstants.N_ADVISORIES):
            # Link advisory issued by agent to current advisory in this loop.
            constrs_to_add.append(
                constrs_manager
                    .create_indicator_constraint(delta_int_adv[adv], 1,
                                                 constrs_manager.get_assignment_constraint(int_advisory_var, adv)))

            # Binary variable for checking whether an acceleration should be chosen
            # non-deterministically.
            [nondet] = constrs_manager.create_binary_variables(1)

            # Binary variable for checking whether an acceleration should not be chosen.
            [det] = constrs_manager.create_binary_variables(1)
            constrs_manager.update()

            # We can only be in either the deterministic or non-deterministic case.
            ## delta[adv] == 1 >> nondet + det == 1
            ## det == 1 >> delta[adv] == 1
            ## nondet == 1 >> delta[adv] == 1
            # Same logic without indicator constraints
            if isinstance(constrs_manager, CustomConstraintsManager):
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([nondet, det, delta_int_adv[adv]], [1, 1, -1]), __ge__, 0))
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([delta_int_adv[adv], det], [1, -1]), __ge__, 0))
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([delta_int_adv[adv], nondet], [1, -1]), __ge__, 0))

            else:
                constrs_to_add.append(nondet + det - delta_int_adv[adv] >= 0)
                constrs_to_add.append(delta_int_adv[adv] - det >= 0)
                constrs_to_add.append(delta_int_adv[adv] - nondet >= 0)

            self.encode_det_nondet(adv, binvars, constrs_manager, constrs_to_add, det,
                                   get_compliance_constraint, grb_vars_next_state, input_state_vars,
                                   joint_action_not_gamma_idx, joint_protocol_not_gamma_indices, nondet, int_acc_idx,
                                   int_adv_idx, int_advisory_var, int_climbrate_idx, 0)
        return constrs_to_add, binvars

    def optimise_adv_own(self, binvars, constrs_manager, constrs_to_add, get_compliance_constraint, grb_vars_next_state,
                         input_state_vars, joint_action_gamma_idx, joint_protocol_indices, next_accelerations_own, own_acc_idx,
                         own_adv_idx, own_advisory_var, own_advisory_var_bounds, own_climbrate_idx):
        adv = own_advisory_var_bounds.get_lower()[0]  # Or upper, doesn't matter.
        if adv == VcasConstants.COC:
            constrs_to_add.append(
                constrs_manager.get_assignment_constraint(
                    own_advisory_var, VcasConstants.COC
                )
            )
            constrs_to_add.append(
                constrs_manager.get_assignment_constraint(
                    grb_vars_next_state[VcasConstants.OWN_ACC_IDX],
                    next_accelerations_own[
                        joint_protocol_indices[joint_action_gamma_idx][0]])
            )
        else:
            constrs_to_add.append(
                constrs_manager.get_assignment_constraint(own_advisory_var, adv)
            )
            # Binary variable for checking whether an acceleration should be chosen
            # non-deterministically.
            [nondet] = constrs_manager.create_binary_variables(1)

            # Binary variable for checking whether an acceleration should not be chosen.
            [det] = constrs_manager.create_binary_variables(1)
            constrs_manager.update()

            if isinstance(constrs_manager, CustomConstraintsManager):
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([nondet, det, 1], [1, 1, -1]), __ge__, 0))
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([1, det], [1, -1]), __ge__, 0))
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([1, nondet], [1, -1]), __ge__, 0))
            else:
                constrs_to_add.append(nondet + det - 1 >= 0)
                constrs_to_add.append(1 - det >= 0)
                constrs_to_add.append(1 - nondet >= 0)
        # Binary variable for checking whether an acceleration should be chosen
        # non-deterministically.
        [nondet] = constrs_manager.create_binary_variables(1)
        # Binary variable for checking whether an acceleration should not be chosen.
        [det] = constrs_manager.create_binary_variables(1)
        constrs_manager.update()
        self.encode_det_nondet(adv, binvars, constrs_manager, constrs_to_add, det,
                               get_compliance_constraint, grb_vars_next_state, input_state_vars,
                               joint_action_gamma_idx, joint_protocol_indices, nondet, own_acc_idx,
                               own_adv_idx, own_advisory_var, own_climbrate_idx, 0)

    def optimise_adv_intruder(self, binvars, constrs_manager, constrs_to_add, get_compliance_constraint, grb_vars_next_state, input_state_vars, int_acc_idx, int_adv_idx, int_advisory_var, int_advisory_var_bounds, int_climbrate_idx, joint_action_gamma_idx, joint_protocol_indices, next_accelerations_int):
        adv = int_advisory_var_bounds.get_lower()[0]  # Or upper, doesn't matter.
        if adv == VcasConstants.COC:
            constrs_to_add.append(
                constrs_manager.get_assignment_constraint(
                    int_advisory_var, VcasConstants.COC
                )
            )
            constrs_to_add.append(
                constrs_manager.get_assignment_constraint(
                    grb_vars_next_state[VcasConstants.INT_ACC_IDX],
                    next_accelerations_int[
                        joint_protocol_indices[joint_action_gamma_idx][1]])
            )
        else:
            constrs_to_add.append(
                constrs_manager.get_assignment_constraint(int_advisory_var, adv)
            )
            # Binary variable for checking whether an acceleration should be chosen
            # non-deterministically.
            [nondet] = constrs_manager.create_binary_variables(1)

            # Binary variable for checking whether an acceleration should not be chosen.
            [det] = constrs_manager.create_binary_variables(1)
            constrs_manager.update()

            if isinstance(constrs_manager, CustomConstraintsManager):
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([nondet, det, 1], [1, 1, -1]), __ge__, 0))
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([1, det], [1, -1]), __ge__, 0))
                constrs_to_add.append(
                    LinearConstraint(WeightedSum([1, nondet], [1, -1]), __ge__, 0))
            else:
                constrs_to_add.append(nondet + det - 1 >= 0)
                constrs_to_add.append(1 - det >= 0)
                constrs_to_add.append(1 - nondet >= 0)
        # Binary variable for checking whether an acceleration should be chosen
        # non-deterministically.
        [nondet] = constrs_manager.create_binary_variables(1)
        # Binary variable for checking whether an acceleration should not be chosen.
        [det] = constrs_manager.create_binary_variables(1)
        constrs_manager.update()
        self.encode_det_nondet(adv, binvars, constrs_manager, constrs_to_add, det,
                               get_compliance_constraint, grb_vars_next_state, input_state_vars,
                               joint_action_gamma_idx, joint_protocol_indices, nondet, int_acc_idx,
                               int_adv_idx, int_advisory_var, int_climbrate_idx, 1)

    def encode_det_nondet(self, adv, binvars, constrs_manager, constrs_to_add, det, get_compliance_constraint,
                                  grb_vars_next_state, input_state_vars, joint_action_idx, joint_protocol_indices,
                                  nondet, acc_idx, adv_idx, advisory_var, climbrate_idx, protocol_indices_inner_idx):
        # Deterministic case only occurs when advisory is the same as the previous one
        # and if the pilot is compliant with the issued advisory, otherwise, we're in
        # the non-deterministic case.
        ## det == 1 >> advisory == input_state_vars[VcasConstants.ADVISORY]
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(
                det, 1, constrs_manager.get_equality_constraint(
                    advisory_var, input_state_vars[adv_idx])))
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(
                det, 1, get_compliance_constraint(adv, input_state_vars[climbrate_idx])))
        # Add equality constraints to non-deterministically choose the acceleration.
        next_accelerations = VcasConstants.ADV_INFO[adv].accelerations
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(
                nondet, 1,
                constrs_manager
                    .get_assignment_constraint(grb_vars_next_state[acc_idx],
                                               next_accelerations[
                                                   joint_protocol_indices[joint_action_idx][protocol_indices_inner_idx]])))
        # If in the deterministic case, maintain the same climbrate.
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(
                det, 1,
                constrs_manager.get_assignment_constraint(grb_vars_next_state[acc_idx], 0)))

    @staticmethod
    def compute_constraints_for_next_state(own_advisory, int_advisory, next_state_vars, input_state_vars, constrs_manager):
        if isinstance(constrs_manager, CustomConstraintsManager):
            return [LinearConstraint(WeightedSum([next_state_vars[VcasConstants.INT_ALT_IDX],
                                                  input_state_vars[VcasConstants.INT_ALT_IDX],
                                                  input_state_vars[VcasConstants.OWN_CLIMBRATE_IDX],
                                                  next_state_vars[VcasConstants.OWN_ACC_IDX],
                                                  input_state_vars[VcasConstants.INT_CLIMBRATE_IDX],
                                                  next_state_vars[VcasConstants.INT_ACC_IDX]],
                                                 [1, -1, 1, 0.5, -1, -0.5]),
                                     __eq__,
                                     0),
                    LinearConstraint(WeightedSum([next_state_vars[VcasConstants.OWN_CLIMBRATE_IDX],
                                                  input_state_vars[VcasConstants.OWN_CLIMBRATE_IDX],
                                                  next_state_vars[VcasConstants.OWN_ACC_IDX]],
                                                 [1, -1, -1]),
                                     __eq__,
                                     0),
                    LinearConstraint(WeightedSum([next_state_vars[VcasConstants.INT_CLIMBRATE_IDX],
                                                  input_state_vars[VcasConstants.INT_CLIMBRATE_IDX],
                                                  next_state_vars[VcasConstants.INT_ACC_IDX]],
                                                 [1, -1, -1]),
                                     __eq__,
                                     0),
                    LinearConstraint(WeightedSum([next_state_vars[VcasConstants.TAU_IDX],
                                                  input_state_vars[VcasConstants.TAU_IDX]],
                                                 [1, -1]),
                                     __eq__,
                                     -1),
                    constrs_manager.get_equality_constraint(next_state_vars[VcasConstants.OWN_ADV_IDX], own_advisory),
                    constrs_manager.get_equality_constraint(next_state_vars[VcasConstants.INT_ADV_IDX], int_advisory)]
        else:
            return [next_state_vars[VcasConstants.INT_ALT_IDX] ==
                    get_intruder_altitude(input_state_vars,
                                          next_state_vars[VcasConstants.OWN_ACC_IDX],
                                          next_state_vars[VcasConstants.INT_ACC_IDX]),
                    next_state_vars[VcasConstants.OWN_CLIMBRATE_IDX] ==
                    get_ownship_vertical_climbrate(input_state_vars, next_state_vars[VcasConstants.OWN_ACC_IDX]),
                    next_state_vars[VcasConstants.INT_CLIMBRATE_IDX] ==
                    get_intruder_vertical_climbrate(input_state_vars, next_state_vars[VcasConstants.INT_ACC_IDX]),
                    next_state_vars[VcasConstants.TAU_IDX] == get_tau(input_state_vars),
                    next_state_vars[VcasConstants.OWN_ADV_IDX] == own_advisory,
                    next_state_vars[VcasConstants.INT_ADV_IDX] == int_advisory]

    @staticmethod
    def compute_next_state_upper_bounds(input_variable_bounds_lower, input_variable_bounds_upper, own_see_bounds, int_see_bounds):
        next_state_upper \
            = np.array([min(VcasConstants.MAX_INT_ALT,
                            input_variable_bounds_upper[VcasConstants.INT_ALT_IDX] -
                            input_variable_bounds_lower[VcasConstants.OWN_CLIMBRATE_IDX] - 0.5 * VcasConstants.MIN_OWN_ACC +
                            input_variable_bounds_upper[VcasConstants.INT_CLIMBRATE_IDX] + 0.5 * VcasConstants.MAX_INT_ACC),
                        min(VcasConstants.MAX_OWN_CLIMBRATE,
                            input_variable_bounds_upper[VcasConstants.OWN_CLIMBRATE_IDX] + VcasConstants.MAX_OWN_ACC),
                        min(VcasConstants.MAX_INT_CLIMBRATE,
                            input_variable_bounds_upper[VcasConstants.INT_CLIMBRATE_IDX] + VcasConstants.MAX_INT_ACC),
                        min(VcasConstants.MAX_TAU, input_variable_bounds_upper[VcasConstants.TAU_IDX] - 1),
                        VcasConstants.MAX_OWN_ACC,
                        VcasConstants.MAX_INT_ACC,
                        VcasConstants.N_ADVISORIES - 1 if own_see_bounds.get_upper() != own_see_bounds.get_lower() else own_see_bounds.get_upper()[0],
                        VcasConstants.N_ADVISORIES - 1 if int_see_bounds.get_upper() != int_see_bounds.get_lower() else int_see_bounds.get_upper()[0]])
        return next_state_upper

    @staticmethod
    def compute_next_state_lower_bounds(input_variable_bounds_lower, input_variable_bounds_upper, own_see_bounds, int_see_bounds):
        next_state_lower \
          = np.array([max(VcasConstants.MIN_INT_ALT, input_variable_bounds_lower[VcasConstants.INT_ALT_IDX] -
                          input_variable_bounds_upper[VcasConstants.OWN_CLIMBRATE_IDX] - 0.5 * VcasConstants.MAX_OWN_ACC +
                          input_variable_bounds_lower[VcasConstants.INT_CLIMBRATE_IDX] + 0.5 * VcasConstants.MIN_INT_ACC),
                      max(VcasConstants.MIN_OWN_CLIMBRATE, input_variable_bounds_lower[VcasConstants.OWN_CLIMBRATE_IDX] +
                          VcasConstants.MIN_OWN_ACC),
                      max(VcasConstants.MIN_INT_CLIMBRATE, input_variable_bounds_lower[VcasConstants.INT_CLIMBRATE_IDX] +
                          VcasConstants.MIN_INT_ACC),
                      max(VcasConstants.MIN_TAU, input_variable_bounds_lower[VcasConstants.TAU_IDX] - 1),
                      VcasConstants.MIN_OWN_ACC,
                      VcasConstants.MIN_INT_ACC,
                      0 if own_see_bounds.get_lower() != own_see_bounds.get_upper() else own_see_bounds.get_lower()[0],
                      0 if int_see_bounds.get_lower() != int_see_bounds.get_upper() else int_see_bounds.get_lower()[0]])
        return next_state_lower
