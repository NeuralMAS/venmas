from src.actors.agents.agent import MultiAgent
from usecases.vcas_multi.gamma_ownship_intruder_two_agent_vcasenv import VcasConstants
from src.utils.constraint import LinearConstraint, WeightedSum
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.constrmanager.custom_constraints_manager import CustomConstraintsManager
from operator import __eq__


def normalise_input(values):
    return (values - VcasConstants.INPUT_MEAN_VALUES) / VcasConstants.INPUT_RANGES


def compute_output_bounds(output_bounds, output_state_vars):
    for j, var in enumerate(output_state_vars):
        l, u = output_bounds[j]
        output_bounds[j] = (min(var.lb, l), max(var.ub, u))


class VcasIntruderAgent(MultiAgent):

    def __str__(self):
        return "Intruder"

    def __repr__(self):
        return str(self)

    def __init__(self, action_space_dim, network_models):
        """
        VcasAgent represents the advisory issuer in the VerticalCAS scenario. It is responsible
        for taking the current state and using an array of neural networks to generate the next
        advisory for the pilot (the environment).
        :param action_space_dim: The amount of possible actions to be produced by the agent.
        """
        self.ACTION_SPACE_DIM = action_space_dim
        self.action_networks = []

        # Number of networks used to generate an advisory.
        self.NUM_NETWORKS = len(network_models)

        # the networks themselves
        self.action_networks = network_models

        # Dimension of the output of the network.
        self.NETWORK_OUTPUT_DIM = VcasConstants.N_ADVISORIES

        # Dimension of the input state of the network. Determines the number of variables from the
        # entire scenario state which make up the actual input state of the network.
        self.NETWORK_INPUT_DIM = VcasConstants.N_STATE_VARS

        super(VcasIntruderAgent, self).__init__()

    def get_constraints_for_obs(self, constrs_manager, input_state_vars):
        qval_estimated = -1
        action_q_gurobi_vars = constrs_manager.create_state_variables(self.NETWORK_OUTPUT_DIM)
        action_q_gurobi_vars_bounds = [(float("inf"), float("-inf")) for _ in range(len(action_q_gurobi_vars))]

        # Initialise a list of constraints to be added to allow only one advisory network to be used at a given time.
        constrs_to_add = []

        # Select the variables which correspond to the network input.
        # The last two variables correspond to the acceleration, which isn't useful for network input.
        useful_raw_network_inputs = input_state_vars[:self.NETWORK_INPUT_DIM]

        # Only difference between ownship and intruder network inputs are here
        raw_bounds = constrs_manager.get_variable_bounds(useful_raw_network_inputs)

        normalised_inputs = []
        normalised_lower_bounds = []
        normalised_upper_bounds = []
        for i, (input_mean, input_range, raw_input_var) \
                in enumerate(zip(VcasConstants.INPUT_MEAN_VALUES,
                                 VcasConstants.INPUT_RANGES,
                                 useful_raw_network_inputs)):
            raw_l, raw_u = raw_bounds.get_dimension_bounds(i)
            normalised_lower_bounds.append((raw_l - input_mean) / input_range)
            normalised_upper_bounds.append((raw_u - input_mean) / input_range)
            [next_var] = constrs_manager \
                .create_state_variables(1,
                                        lbs=[normalised_lower_bounds[i]],
                                        ubs=[normalised_upper_bounds[i]])
            if isinstance(constrs_manager, CustomConstraintsManager):
                constrs_to_add.append(
                    LinearConstraint(
                        WeightedSum([next_var, raw_input_var], [1, -1/input_range]),
                                     __eq__,
                                     -input_mean/input_range)
                )
            else:
                constrs_to_add.append(
                    next_var == (raw_input_var - input_mean) / input_range
                )
            normalised_inputs.append(next_var)

        constrs_manager.update()
        constrs_manager.add_variable_bounds(normalised_inputs,
                                            HyperRectangleBounds(normalised_lower_bounds, normalised_upper_bounds))

        # Intruder agent takes as input a linear transformation of the input of the ownship agent.
        normalised_raw_bounds = constrs_manager.get_variable_bounds(normalised_inputs)
        int_input_1_lo, int_input_1_hi = normalised_raw_bounds.get_dimension_bounds(0)
        int_input_2_lo, int_input_2_hi = normalised_raw_bounds.get_dimension_bounds(1)
        int_input_3_lo, int_input_3_hi = normalised_raw_bounds.get_dimension_bounds(2)
        int_input_4_lo, int_input_4_hi = normalised_raw_bounds.get_dimension_bounds(3)

        int_inputs_lbs = [-int_input_1_hi, int_input_3_lo, int_input_2_lo, int_input_4_lo]
        int_inputs_ubs = [-int_input_1_lo, int_input_3_hi, int_input_2_hi, int_input_4_hi]
        int_nw_input_vars = constrs_manager.create_state_variables(4,
                                                                   lbs=int_inputs_lbs,
                                                                   ubs=int_inputs_ubs)

        if isinstance(constrs_manager, CustomConstraintsManager):
            constrs_to_add.extend([
                LinearConstraint(
                    WeightedSum([int_nw_input_vars[0], normalised_inputs[0]], [1, 1]),
                    __eq__,
                    0),
                LinearConstraint(
                    WeightedSum([int_nw_input_vars[1], normalised_inputs[2]], [1, -1]),
                    __eq__,
                    0),
                LinearConstraint(
                    WeightedSum([int_nw_input_vars[2], normalised_inputs[1]], [1, -1]),
                    __eq__,
                    0),
                LinearConstraint(
                    WeightedSum([int_nw_input_vars[3], normalised_inputs[3]], [1, -1]),
                    __eq__,
                    0),
            ]
            )
        else:
            constrs_to_add.extend([
                int_nw_input_vars[0] == -normalised_inputs[0],
                int_nw_input_vars[1] == normalised_inputs[2],
                int_nw_input_vars[2] == normalised_inputs[1],
                int_nw_input_vars[3] == normalised_inputs[3],
            ]
            )

        constrs_manager.update()
        constrs_manager.add_variable_bounds(int_nw_input_vars,
                                            HyperRectangleBounds(int_inputs_lbs, int_inputs_ubs))

        # optimisation for narrowing down the range of advisories to be considered
        input_bounds = constrs_manager.get_variable_bounds(input_state_vars)
        own_adv_bounds_lower = input_bounds.get_lower()[VcasConstants.OWN_ADV_IDX]
        own_adv_bounds_upper = input_bounds.get_upper()[VcasConstants.OWN_ADV_IDX]

        delta_lower_bounds = [0 for _ in range(self.NUM_NETWORKS)]
        delta_upper_bounds = [1 for _ in range(self.NUM_NETWORKS)]

        if own_adv_bounds_lower == own_adv_bounds_upper:
            delta_upper_bounds = [0 for _ in range(self.NUM_NETWORKS)]
            delta_lower_bounds[int(own_adv_bounds_lower)] = 1
            delta_upper_bounds[int(own_adv_bounds_lower)] = 1
        else:
            for i in range(self.NUM_NETWORKS):
                if own_adv_bounds_lower > i or own_adv_bounds_upper < i:
                    delta_upper_bounds[i] = 0

        # binary variables for the network constraints corresponding to the previous advisory
        delta = constrs_manager.create_binary_variables(self.NUM_NETWORKS, lbs=delta_lower_bounds, ubs=delta_upper_bounds)
        constrs_to_add.append(constrs_manager.get_sum_constraint(delta, 1))

        constrs_manager.update()

        for adv in [VcasConstants.COC,
                    VcasConstants.DNC, VcasConstants.DND, VcasConstants.DES1500, VcasConstants.CL1500,
                    VcasConstants.SDES1500, VcasConstants.SCL1500, VcasConstants.SDES2500, VcasConstants.SCL2500]:

            # Skip the current advisory if delta[adv] is trivially False
            if delta[adv].ub == 0:
                continue

            # Add constraints for all advisory networks.
            q_vars, network_constrs = constrs_manager \
                .get_network_constraints(self.action_networks[adv].layers, int_nw_input_vars)

            # q_vars = constrs_manager.create_state_variables(self.NETWORK_OUTPUT_DIM)
            # network_constrs = [q == -2.2 for q in q_vars[:adv]+q_vars[adv+1:]] + [q_vars[adv] == adv]

            # Add constraints linking the advisory to the specific network to be used.
            constrs_to_add.append(
                constrs_manager.create_indicator_constraint(delta[adv], 1,
                                                            constrs_manager.get_assignment_constraint(
                                                                input_state_vars[VcasConstants.OWN_ADV_IDX],
                                                                adv))
            )
            for constr in network_constrs:
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(delta[adv], 1, constr))

            # Add constraints linking the q-values from the network of the chosen advisory to the
            # q-values output by the agent.
            for q_value in range(self.NETWORK_OUTPUT_DIM):
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(delta[adv], 1,
                                                                constrs_manager.get_equality_constraint(
                                                                    action_q_gurobi_vars[q_value],
                                                                    q_vars[q_value])))

            # update the bounds for action_q_gurobi_vars
            compute_output_bounds(action_q_gurobi_vars_bounds, q_vars)

        # Add the bounds for action_q_gurobi_vars to constra_manager
        action_q_gurobi_vars_lower, action_q_gurobi_vars_upper = zip(*action_q_gurobi_vars_bounds)
        constrs_manager.add_variable_bounds(action_q_gurobi_vars, HyperRectangleBounds(action_q_gurobi_vars_lower, action_q_gurobi_vars_upper))

        # if qval_estimated == -1:
        # Add the encoding of argmax for the computed q-values.
        action_grb_vars, action_grb_constraints = \
            constrs_manager.get_argmax_constraints(action_q_gurobi_vars, use_q_bounds=True  )
        constrs_to_add.extend(action_grb_constraints)


        # Return a single integer variable as the output of argmax.
        l = len(action_grb_vars)
        integer_argmax_vars, integer_argmax_constrs = \
            constrs_manager.get_argmax_index_constraints(action_grb_vars)
        constrs_manager.add_variable_bounds([integer_argmax_vars], HyperRectangleBounds([integer_argmax_vars.lb],
                                                                                        [integer_argmax_vars.ub]))

        constrs_to_add.extend(integer_argmax_constrs)

        return [integer_argmax_vars], constrs_to_add


class VcasOwnshipAgent(MultiAgent):

    def __str__(self):
        return "Ownship"

    def __repr__(self):
        return str(self)

    def __init__(self, action_space_dim, network_models):
        """
        VcasAgent represents the advisory issuer in the VerticalCAS scenario. It is responsible
        for taking the current state and using an array of neural networks to generate the next
        advisory for the pilot (the environment).
        :param action_space_dim: The amount of possible actions to be produced by the agent.
        """
        self.ACTION_SPACE_DIM = action_space_dim
        self.action_networks = []

        # Number of networks used to generate an advisory.
        self.NUM_NETWORKS = len(network_models)

        # the networks themselves
        self.action_networks = network_models

        # Dimension of the output of the network.
        self.NETWORK_OUTPUT_DIM = VcasConstants.N_ADVISORIES

        # Dimension of the input state of the network. Determines the number of variables from the
        # entire scenario state which make up the actual input state of the network.
        self.NETWORK_INPUT_DIM = VcasConstants.N_STATE_VARS

        super(VcasOwnshipAgent, self).__init__()

    def get_constraints_for_obs(self, constrs_manager, input_state_vars):
        qval_estimated = -1
        action_q_gurobi_vars = constrs_manager.create_state_variables(self.NETWORK_OUTPUT_DIM)
        action_q_gurobi_vars_bounds = [(float("inf"), float("-inf")) for _ in range(len(action_q_gurobi_vars))]

        # Initialise a list of constraints to be added to allow only one advisory network to be used at a given time.
        constrs_to_add = []

        # Select the variables which correspond to the network input.
        # The last two variables correspond to the acceleration, which isn't useful for network input.
        useful_raw_network_inputs = input_state_vars[:self.NETWORK_INPUT_DIM]
        raw_network_inputs = useful_raw_network_inputs
        raw_bounds = constrs_manager.get_variable_bounds(raw_network_inputs)

        normalised_inputs = []
        normalised_lower_bounds = []
        normalised_upper_bounds = []
        for i, (input_mean, input_range, raw_input_var) \
                in enumerate(zip(VcasConstants.INPUT_MEAN_VALUES,
                                 VcasConstants.INPUT_RANGES,
                                 raw_network_inputs)):
            raw_l, raw_u = raw_bounds.get_dimension_bounds(i)
            normalised_lower_bounds.append((raw_l - input_mean) / input_range)
            normalised_upper_bounds.append((raw_u - input_mean) / input_range)
            [next_var] = constrs_manager \
                .create_state_variables(1,
                                        lbs=[normalised_lower_bounds[i]],
                                        ubs=[normalised_upper_bounds[i]])
            if isinstance(constrs_manager, CustomConstraintsManager):
                constrs_to_add.append(
                    LinearConstraint(
                        WeightedSum([next_var, raw_input_var], [1, -1/input_range]),
                                     __eq__,
                                     -input_mean/input_range)
                )
            else:
                constrs_to_add.append(
                    next_var == (raw_input_var - input_mean) / input_range
                )
            normalised_inputs.append(next_var)

        constrs_manager.update()
        constrs_manager.add_variable_bounds(normalised_inputs,
                                            HyperRectangleBounds(normalised_lower_bounds, normalised_upper_bounds))

        # optimisation for narrowing down the range of advisories to be considered
        input_bounds = constrs_manager.get_variable_bounds(input_state_vars)
        own_adv_bounds_lower = input_bounds.get_lower()[VcasConstants.OWN_ADV_IDX]
        own_adv_bounds_upper = input_bounds.get_upper()[VcasConstants.OWN_ADV_IDX]

        delta_lower_bounds = [0 for _ in range(self.NUM_NETWORKS)]
        delta_upper_bounds = [1 for _ in range(self.NUM_NETWORKS)]

        if own_adv_bounds_lower == own_adv_bounds_upper:
            delta_upper_bounds = [0 for _ in range(self.NUM_NETWORKS)]
            delta_lower_bounds[int(own_adv_bounds_lower)] = 1
            delta_upper_bounds[int(own_adv_bounds_lower)] = 1
        else:
            for i in range(self.NUM_NETWORKS):
                if own_adv_bounds_lower > i or own_adv_bounds_upper < i:
                    delta_upper_bounds[i] = 0

        # binary variables for the network constraints corresponding to the previous advisory
        delta = constrs_manager.create_binary_variables(self.NUM_NETWORKS, lbs=delta_lower_bounds, ubs=delta_upper_bounds)
        constrs_to_add.append(constrs_manager.get_sum_constraint(delta, 1))

        constrs_manager.update()

        # if own_adv_bounds_lower == own_adv_bounds_upper and not isinstance(constrs_manager, CustomConstraintsManagerWithBounds):
        #     adv = int(own_adv_bounds_upper)  # Or lower - doesn't really matter.
        #     q_vars, network_constrs = constrs_manager \
        #         .get_network_constraints(self.action_networks[adv].layers, normalised_inputs)
        #
        #     constrs_to_add.append(
        #         constrs_manager.get_assignment_constraint(
        #             input_state_vars[VcasConstants.OWN_ADV_IDX],
        #             adv
        #         )
        #     )
        #
        #     for constr in network_constrs:
        #         constrs_to_add.append(
        #             constr
        #         )
        #
        #     # TODO: Improve to handle more cases
        #     amax_estimate_u = np.argmax(constrs_manager.get_variable_bounds(q_vars).get_upper())
        #     amax_estimate_l = np.argmax(constrs_manager.get_variable_bounds(q_vars).get_lower())
        #     if amax_estimate_l == amax_estimate_u:  # Bounds tight enough to find the right answer immediately.
        #         qval_estimated = amax_estimate_u
        #     else:
        #         for q_value in range(self.NETWORK_OUTPUT_DIM):
        #             constrs_to_add.append(
        #                 constrs_manager.get_equality_constraint(
        #                     action_q_gurobi_vars[q_value],
        #                     q_vars[q_value]
        #                 )
        #             )
        #
        # else:
        for adv in [VcasConstants.COC,
                    VcasConstants.DNC, VcasConstants.DND, VcasConstants.DES1500, VcasConstants.CL1500,
                    VcasConstants.SDES1500, VcasConstants.SCL1500, VcasConstants.SDES2500, VcasConstants.SCL2500]:

            # Skip the current advisory if delta[adv] is trivially False
            if delta[adv].ub == 0:
                continue

            # Add constraints for all advisory networks.
            q_vars, network_constrs = constrs_manager \
                .get_network_constraints(self.action_networks[adv].layers, normalised_inputs)

            # q_vars = constrs_manager.create_state_variables(self.NETWORK_OUTPUT_DIM)
            # network_constrs = [q == -2.2 for q in q_vars[:adv]+q_vars[adv+1:]] + [q_vars[adv] == adv]

            # Add constraints linking the advisory to the specific network to be used.
            constrs_to_add.append(
                constrs_manager.create_indicator_constraint(delta[adv], 1,
                                                            constrs_manager.get_assignment_constraint(
                                                                input_state_vars[VcasConstants.OWN_ADV_IDX],
                                                                adv))
            )
            for constr in network_constrs:
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(delta[adv], 1, constr))

            # Add constraints linking the q-values from the network of the chosen advisory to the
            # q-values output by the agent.
            for q_value in range(self.NETWORK_OUTPUT_DIM):
                constrs_to_add.append(
                    constrs_manager.create_indicator_constraint(delta[adv], 1,
                                                                constrs_manager.get_equality_constraint(
                                                                    action_q_gurobi_vars[q_value],
                                                                    q_vars[q_value])))

            # update the bounds for action_q_gurobi_vars
            compute_output_bounds(action_q_gurobi_vars_bounds, q_vars)

        # Add the bounds for action_q_gurobi_vars to constra_manager
        action_q_gurobi_vars_lower, action_q_gurobi_vars_upper = zip(*action_q_gurobi_vars_bounds)
        constrs_manager.add_variable_bounds(action_q_gurobi_vars, HyperRectangleBounds(action_q_gurobi_vars_lower, action_q_gurobi_vars_upper))

        # Add the encoding of argmax for the computed q-values.
        # if qval_estimated == -1:
        action_grb_vars, action_grb_constraints = constrs_manager.get_argmax_constraints(action_q_gurobi_vars, use_q_bounds=True)
        constrs_to_add.extend(action_grb_constraints)

        # Return a single integer variable as the output of argmax.
        l = len(action_grb_vars)
        integer_argmax_vars, integer_argmax_constrs = constrs_manager.get_argmax_index_constraints(action_grb_vars)
        constrs_manager.add_variable_bounds([integer_argmax_vars], HyperRectangleBounds([integer_argmax_vars.lb],
                                                                                        [integer_argmax_vars.ub]))
        constrs_to_add.extend(integer_argmax_constrs)

        return [integer_argmax_vars], constrs_to_add
        # else:
        #     if isinstance(constrs_manager, CustomConstraintsManagerWithBounds):
        #         o = constrs_manager._add_integer_variable(qval_estimated, qval_estimated)
        #         constrs_manager.add_variable_bounds([o],
        #                                             HyperRectangleBounds([qval_estimated],
        #                                                                  [qval_estimated]))
        #         constrs_manager.update()
        #     else:
        #         o = constrs_manager.create_integer_variable(qval_estimated, qval_estimated)
        #         constrs_manager.add_variable_bounds([o],
        #                                             HyperRectangleBounds([qval_estimated],
        #                                                                  [qval_estimated]))
        #         constrs_manager.update()
        #     constrs_to_add.append(
        #         constrs_manager.get_equality_constraint(o, qval_estimated)
        #     )
        #
        #     return [o], constrs_to_add, binvars
