from src.actors.agents.agent import Agent
from usecases.vcas.vcasenv import VcasConstants, get_advisories_delta_bounds
from src.utils.utils import get_widest_bounds
from src.verification.bounds.bounds import HyperRectangleBounds


def normalise_input(values):
    return (values - VcasConstants.INPUT_MEAN_VALUES) / VcasConstants.INPUT_RANGES


class VcasAgent(Agent):

    def __init__(self, network_models):
        """
        VcasAgent represents the advisory issuer in the VerticalCAS scenario. It is responsible
        for taking the current state and using an array of neural networks to generate the next
        advisory for the pilot (the environment).
        :param network_models: The internal representation of the 9 models
        """
        # The dimensionality of the action space,
        # it is 1 as the action is the advisory.
        self.ACTION_SPACE_DIM = 1

        # Number of networks used to generate an advisory.
        self.NUM_NETWORKS = len(network_models)

        # the networks themselves
        self.action_networks = network_models

        # Dimension of the output of the network.
        self.NETWORK_OUTPUT_DIM = VcasConstants.N_ADVISORIES

        # Dimension of the input state of the network. Determines the number of variables from the
        # entire scenario state which make up the actual input state of the network.
        self.NETWORK_INPUT_DIM = VcasConstants.N_STATE_VARS

        super(VcasAgent, self).__init__()

    def get_constraints_for_action(self, constrs_manager, input_state_vars):
        """
        Create Gurobi constraints for performing an action. Constraints are only added to the model by
        the caller, to reduce side-effects in this function.
        :param constrs_manager: Manager of Gurobi constraints.
        :param input_state_vars: Gurobi variables representing the input state passed to the agent.
        :return: Singleton list of Gurobi variables.
        :side-effects: Modifies constraints manager when adding variables.
        """

        # Initialise a list of constraints to be added to allow only one advisory network to be used at a given time.
        constrs_to_add = []

        # Normalise the state variables to feed to the network
        normalised_inputs, normalised_constrs = self.get_normalised_inputs(constrs_manager, input_state_vars)
        constrs_to_add.extend(normalised_constrs)

        # Get delta variables for previous advisories with optimal bounds
        # (that depend on the bounds of the previous advisory)
        delta = self.get_delta_vars_for_advisories_with_optimal_bounds(constrs_manager, input_state_vars)

        # Initialise a list of constraints to be added to allow only one advisory network to be used at a given time.
        constrs_to_add.append(constrs_manager.get_sum_constraint(delta, 1))

        constrs_manager.update()

        action_q_vars = constrs_manager.create_state_variables(self.NETWORK_OUTPUT_DIM)
        action_q_vars_bounds = [(float("inf"), float("-inf")) for _ in range(len(action_q_vars))]

        for adv in [VcasConstants.COC,
                    VcasConstants.DNC, VcasConstants.DND, VcasConstants.DES1500, VcasConstants.CL1500,
                    VcasConstants.SDES1500, VcasConstants.SCL1500, VcasConstants.SDES2500, VcasConstants.SCL2500]:

            # Skip the current advisory if delta[adv] is trivially False
            if delta[adv].ub == 0:
                continue

            # Add constraints for the current advisory network.
            q_vars, network_constrs = constrs_manager.get_network_constraints(self.action_networks[adv].layers, normalised_inputs)

            # Add constraints linking the advisory to the specific network to be used.
            constrs_to_add.append(
                constrs_manager.create_indicator_constraint(
                    delta[adv], 1,
                    constrs_manager.get_assignment_constraint(input_state_vars[VcasConstants.ADVISORY], adv))
            )
            for constr in network_constrs:
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(delta[adv], 1, constr)
                )

            # Add constraints linking the q-values from the network of the chosen advisory to the
            # q-values output by the agent.
            for q_value in range(self.NETWORK_OUTPUT_DIM):
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(
                        delta[adv], 1,
                        constrs_manager.get_equality_constraint(action_q_vars[q_value], q_vars[q_value]))
                )

            # update the bounds for percept_q_vars
            get_widest_bounds(action_q_vars_bounds, q_vars)

        # Add the bounds for action_q_vars to constr_manager
        action_q_vars_lower, action_q_vars_upper = zip(*action_q_vars_bounds)
        constrs_manager.add_variable_bounds(action_q_vars, HyperRectangleBounds(action_q_vars_lower, action_q_vars_upper))

        # Add the encoding of argmax for the computed q-values.
        action_vars, action_constraints = constrs_manager.get_argmax_constraints(action_q_vars, use_q_bounds=True)
        constrs_to_add.extend(action_constraints)

        # Return a single integer variable as the output of argmax.
        integer_argmax_var, integer_argmax_constrs = constrs_manager.get_argmax_index_constraints(action_vars)
        constrs_manager.add_variable_bounds([integer_argmax_var],
                                            HyperRectangleBounds([integer_argmax_var.lb], [integer_argmax_var.ub]))
        constrs_to_add.extend(integer_argmax_constrs)

        return [integer_argmax_var], constrs_to_add

    def get_normalised_inputs(self, constrs_manager, input_state_vars):
        constrs = []

        # Select the variables which correspond to the network input.
        raw_network_inputs = input_state_vars[:self.NETWORK_INPUT_DIM]
        raw_bounds = constrs_manager.get_variable_bounds(raw_network_inputs)

        normalised_inputs = []
        normalised_lower_bounds = []
        normalised_upper_bounds = []
        for i, (input_mean, input_range, raw_input_var) in \
                enumerate(zip(VcasConstants.INPUT_MEAN_VALUES, VcasConstants.INPUT_RANGES, raw_network_inputs)):
            raw_l, raw_u = raw_bounds.get_dimension_bounds(i)
            normalised_lower_bounds.append((raw_l - input_mean) / input_range)
            normalised_upper_bounds.append((raw_u - input_mean) / input_range)
            [next_var] = constrs_manager.create_state_variables(1,
                                                                lbs=[normalised_lower_bounds[i]],
                                                                ubs=[normalised_upper_bounds[i]])
            # next_var == (raw_input_var - input_mean) / input_range
            constrs.append(
                constrs_manager.get_linear_constraint([next_var, raw_input_var], [1, -1/input_range], -input_mean/input_range)
            )
            normalised_inputs.append(next_var)

        constrs_manager.update()
        constrs_manager.add_variable_bounds(normalised_inputs,
                                            HyperRectangleBounds(normalised_lower_bounds, normalised_upper_bounds))

        return normalised_inputs, constrs

    def get_delta_vars_for_advisories_with_optimal_bounds(self, constrs_manager, input_state_vars):
        # optimisation for narrowing down the range of advisories to be considered
        input_bounds = constrs_manager.get_variable_bounds(input_state_vars)

        delta_lower_bounds, delta_upper_bounds = get_advisories_delta_bounds(
            input_bounds.get_dimension_bounds(VcasConstants.ADVISORY))

        # binary variables for the network constraints corresponding to the previous advisory
        delta = constrs_manager.create_binary_variables(VcasConstants.N_ADVISORIES,
                                                        lbs=delta_lower_bounds,
                                                        ubs=delta_upper_bounds)
        return delta

