from src.actors.agents.agent import TemplateAgent
from src.verification.bounds.bounds import HyperRectangleBounds

from operator import __le__, __ge__, __eq__


class GuardingConstants:
    EXPIRED_OBSERVATION = 0
    REST_OBSERVATION = 1
    VOLUNTEER_GUARD_OBSERVATION = 2

    EXPIRED_ACTION = 0
    REST_ACTION = 1
    GUARD_ACTION = 2

    GUARDING_REWARD = -2
    RESTING_REWARD = 1
    UNGUARDED_REWARD = -3

    EXPIRED_HEALTH_POINTS = 0
    MAX_HEALTH_POINTS = 4

    AGENT_STATE_DIMENSIONS = 2
    # The index for the health value in a state
    HEALTH_IDX = 0
    PERCEPT_IDX = 1

    # the branching factor of the agent.
    # When volunteering to guard, the agent has a choice of actions: rest or guard
    # In other cases there are always exactly one action to choose from
    AGENT_MAX_ACTIONS = 2

    # Maximum absolute value of health that can be obtained, to be used for encoding max and min constraints
    BIG_M_CONSTANT = MAX_HEALTH_POINTS - UNGUARDED_REWARD


class GuardingAgent(TemplateAgent):

    def __init__(self, obs_network_model):
        """
        GuardingAgent represents an agent template in the guarding game.

        In the game, there is a colony of agents. The colony
        needs to be guarded by exactly one guard. Guarding duty
        costs the guard some health, Gp, while those who are resting
        improve their health by Rr, where the health is limited by a
        maximum value Mh. If no one is guarding the colony, then
        all the agents in the colony lose some health, Up. When an
        agent does not have any health left, it expires.
        :param network_models: The internal representation of the 9 models
        """
        # The dimensionality of the action space,
        # it is 1 as the action is the advisory.
        self.ACTION_SPACE_DIM = 1

        # the network representing the observation function
        self.observation = obs_network_model

        # Dimension of the output of the network.
        # 0 - rest only
        # 1 - volunteer to guard (two available actions, rest and guard)
        # There is also an expired action, but it is not part of the NN output
        # We will hardcode it for the inputs where the health points are <= 0
        self.NETWORK_OUTPUT_DIM = 2

        # Dimension of the input state of the network. Determines the number of variables from the
        # entire scenario state which make up the actual input state of the network.
        #
        # The input to the neural network is health_points
        self.NETWORK_INPUT_DIM = 1

        super(GuardingAgent, self).__init__()

    def get_constraints_for_obs(self, constrs_manager, input_state_vars):
        """
        Create Gurobi constraints for performing an observation on the env.
        Get the constraints for the network itself, set constraints to
        determine the percept output by the agent. Will need to compute the
        percept of the agent based on the Q-value with highest value.
        :param constrs_manager: Constraint manager handling all constraints.
        :param input_state_vars: Set of variables representing the state that
        is input to the agent's protocol function.
        :side-effects: May modify constrs_manager.
        :return:
            variables representing action taken,
            the constraints and
            the binary variables used in indicator constraints.
        """

        # Initialise a list of constraints to be added to allow only one advisory network to be used at a given time.
        constrs_to_add = []


        # Select the variables which correspond to the network input.
        # It is only the first component, encoding the health points
        network_input_vars = input_state_vars[:self.NETWORK_INPUT_DIM]


        ###############################################################################################
        ### Compute the output of the neural network to know the bounds on the observation variable ###
        ### before it is created                                                                    ###
        ###############################################################################################

        # Normalise the state variables to feed to the network
        normalised_inputs, normalised_constrs = self._get_normalised_inputs(constrs_manager, network_input_vars)
        constrs_to_add.extend(normalised_constrs)

        # Compute constraints for the observation neural network
        q_vars, network_constrs = constrs_manager.get_network_constraints(self.observation.layers, normalised_inputs)
        # Add the encoding of argmax for the computed q-values.
        one_hot_argmax_vars, argmax_constrs = constrs_manager.get_argmax_constraints(q_vars, use_q_bounds=True)
        # Return a single integer variable as the output of argmax.
        integer_argmax_var, integer_argmax_constrs = constrs_manager.get_argmax_index_constraints(one_hot_argmax_vars)
        # constrs_manager.add_variable_bounds([integer_argmax_var],
        #                                     HyperRectangleBounds([integer_argmax_var.lb], [integer_argmax_var.ub]))


        # The bounds for the observation and the binary variable expired
        obs_lb = 0
        obs_ub = 2
        expired_lb = 0
        expired_ub = 1

        health_lb, health_ub = constrs_manager.get_variable_bounds(network_input_vars).get_dimension_bounds(0)
        # If we know that the health points is positive, then the agent is not expired
        # So we can set tighter bounds to the observation variable
        if health_lb > GuardingConstants.EXPIRED_HEALTH_POINTS:
            expired_ub = 0

            # This takes into account the fact that when not expired,
            # the observation result is integer_argmax_var + 1
            obs_lb = integer_argmax_var.lb + 1
            obs_ub = integer_argmax_var.ub + 1
        elif health_ub <= GuardingConstants.EXPIRED_HEALTH_POINTS:
            obs_ub = 0
            expired_lb = 1


        ### Create the variables observation and expired ###
        # Binary variable for checking whether the agent is expired (health_points is non-positive)
        [expired] = constrs_manager.create_binary_variables(1, lbs=[expired_lb], ubs=[expired_ub])

        # Integer variable for the resulting observation
        # 0 - Expired
        # 1 - Rest
        # 2 - (volunteer to) Guard
        integer_obs_var = constrs_manager.create_integer_variable(lb=obs_lb, ub=obs_ub)
        constrs_manager.add_variable_bounds([integer_obs_var], HyperRectangleBounds([obs_lb], [obs_ub]))

        # If expired, then health_points should be non-positive and the observation should be 0
        constrs_to_add.extend([
            constrs_manager.create_indicator_constraint(expired, 1,
                                                        constrs_manager.get_le_constraint(
                                                            network_input_vars[0],
                                                            GuardingConstants.EXPIRED_HEALTH_POINTS)),
            constrs_manager.create_indicator_constraint(expired, 1,
                                                        constrs_manager.get_assignment_constraint(
                                                            integer_obs_var,
                                                            GuardingConstants.EXPIRED_OBSERVATION))
        ])

        #########################################################################
        # Otherwise, health_points should be at least 1                         #
        # and we add the constraints for computing the observation using the NN #
        #########################################################################
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(expired, 0,
                                                        constrs_manager.get_ge_constraint(
                                                            network_input_vars[0],
                                                            GuardingConstants.EXPIRED_HEALTH_POINTS + 1))
        )

        # argmax_constrs would typically have indicator constraints and one sum constraint.
        # integer_argmax_constrs would typically have indicator constraints only.
        # We cannot nest indicator constraints, so we add them without a guard (expired=0).
        # If expired = 1, then they would be enforced, but not really used anywhere.
        # If expired = 0, then they would be enforced and used.
        constrs_to_add.extend([constrs_manager.create_indicator_constraint(expired, 0, constr)
                               for constr in network_constrs] + argmax_constrs + integer_argmax_constrs)

        # The final observation should be integer_argmax_var + 1
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(expired, 0,
                                                        constrs_manager.get_linear_constraint(
                                                            [integer_obs_var, integer_argmax_var], [1, -1], 1))
        )

        return [integer_obs_var], constrs_to_add

    def _get_normalised_inputs(self, constrs_manager, network_input_vars):
        constrs = []

        raw_bounds = constrs_manager.get_variable_bounds(network_input_vars)
        raw_l, raw_u = raw_bounds.get_dimension_bounds(0)

        normalised_lb = raw_l / GuardingConstants.MAX_HEALTH_POINTS
        normalised_ub = raw_u / GuardingConstants.MAX_HEALTH_POINTS
        normalised_inputs = constrs_manager.create_state_variables(1, lbs=[normalised_lb], ubs=[normalised_ub])

        # next_var == (raw_input_var - input_mean) / input_range
        constrs.append(
            constrs_manager.get_linear_constraint([normalised_inputs[0], network_input_vars[0]], [1, -1/GuardingConstants.MAX_HEALTH_POINTS], 0)
        )

        constrs_manager.update()
        constrs_manager.add_variable_bounds(normalised_inputs,
                                            HyperRectangleBounds([normalised_lb], [normalised_ub]))

        return normalised_inputs, constrs

    def get_constraints_for_protocol(self, constrs_manager, input_state_vars):
        """
        Create constraints for computing the available actions in the given state.

        Constraints are only added to the model by
        the caller, to reduce side-effects in this function.
        :param constrs_manager: Manager of Gurobi constraints.
        :param input_state_vars: Gurobi variables representing the input state passed to the agent.
        :return:
            a list of action vars and a list of action constraints,
            where i-th element corresponds to i-th possible action.
        :side-effects: Modifies constraints manager when adding variables.
        """

        percept_var = input_state_vars[GuardingConstants.PERCEPT_IDX]
        percept_lb, percept_ub = constrs_manager.get_variable_bounds([percept_var]).get_dimension_bounds(0)

        constrs_to_add = []

        expired_lb = need_rest_lb = volunteered_lb = 0
        expired_ub = need_rest_ub = volunteered_ub = 1
        action1_lb = action2_lb = GuardingConstants.EXPIRED_ACTION
        action1_ub = action2_ub = GuardingConstants.GUARD_ACTION

        if percept_ub <= GuardingConstants.EXPIRED_OBSERVATION:
            expired_lb = 1
            need_rest_ub = 0
            volunteered_ub = 0
            action1_ub = action2_ub = GuardingConstants.EXPIRED_ACTION
        elif percept_lb >= GuardingConstants.REST_OBSERVATION and percept_ub <= GuardingConstants.REST_OBSERVATION:
            expired_ub = 0
            need_rest_lb = 1
            volunteered_ub = 0
            action1_lb = action2_lb = action1_ub = action2_ub = GuardingConstants.REST_ACTION
        elif percept_lb >= GuardingConstants.VOLUNTEER_GUARD_OBSERVATION:
            expired_ub = 0
            need_rest_ub = 0
            volunteered_lb = 1
            action1_lb = action1_ub = GuardingConstants.REST_ACTION
            action2_lb = action2_ub = GuardingConstants.GUARD_ACTION

        # Binary variable for checking whether the percept is expired, need rest or volunteered to guard
        [expired, need_rest, volunteered] = constrs_manager.create_binary_variables(3, lbs=[expired_lb, need_rest_lb, volunteered_lb],
                                                                                    ubs=[expired_ub, need_rest_ub, volunteered_ub])

        constrs_to_add.extend([
            constrs_manager.create_indicator_constraint(
                expired, 1,
                constrs_manager.get_assignment_constraint(percept_var, GuardingConstants.EXPIRED_OBSERVATION)),
            constrs_manager.create_indicator_constraint(
                need_rest, 1,
                constrs_manager.get_assignment_constraint(percept_var, GuardingConstants.REST_OBSERVATION)),
            constrs_manager.create_indicator_constraint(
                volunteered, 1,
                constrs_manager.get_assignment_constraint(percept_var, GuardingConstants.VOLUNTEER_GUARD_OBSERVATION)),
            constrs_manager.get_sum_constraint([expired, need_rest, volunteered], 1)
        ])


        # The branching factor is 2, so we return two sets of constraints and two sets of action vars
        constrs1 = [] + constrs_to_add
        constrs2 = [] + constrs_to_add
        action1 = constrs_manager.create_integer_variable(lb=action1_lb, ub=action1_ub)
        action2 = constrs_manager.create_integer_variable(lb=action2_lb, ub=action2_ub)
        constrs_manager.add_variable_bounds([action1, action2], HyperRectangleBounds([action1_lb, action2_lb],
                                                                                     [action1_ub, action2_ub]))

        constrs1.extend([
            constrs_manager.create_indicator_constraint(
                expired, 1,
                constrs_manager.get_assignment_constraint(action1, GuardingConstants.EXPIRED_ACTION)),
            constrs_manager.create_indicator_constraint(
                need_rest, 1,
                constrs_manager.get_assignment_constraint(action1, GuardingConstants.REST_ACTION)),
            constrs_manager.create_indicator_constraint(
                volunteered, 1,
                constrs_manager.get_assignment_constraint(action1, GuardingConstants.REST_ACTION))
        ])
        constrs2.extend([
            constrs_manager.create_indicator_constraint(
                expired, 1,
                constrs_manager.get_assignment_constraint(action2, GuardingConstants.EXPIRED_ACTION)),
            constrs_manager.create_indicator_constraint(
                need_rest, 1,
                constrs_manager.get_assignment_constraint(action2, GuardingConstants.REST_ACTION)),
            constrs_manager.create_indicator_constraint(
                volunteered, 1,
                constrs_manager.get_assignment_constraint(action2, GuardingConstants.GUARD_ACTION))
        ])

        return [[action1], [action2]], [constrs1, constrs2]

    def get_constraints_for_transition(self, constrs_manager, state_vars,
                                       own_action_vars, joint_action_vars, env_action_vars):

        action_var = own_action_vars[0]
        action_lb, action_ub = constrs_manager.get_variable_bounds([action_var]).get_dimension_bounds(0)
        health_var = state_vars[GuardingConstants.HEALTH_IDX]
        health_lb, health_ub = constrs_manager.get_variable_bounds([health_var]).get_dimension_bounds(0)

        constrs_to_add = []


        # Detecting if at least one agent guards or nobody guards
        # Assumption here that action is a single variable
        action_bounds = constrs_manager.get_variable_bounds(joint_action_vars)
        at_least_one_guarding = False
        for action_number, action in enumerate(joint_action_vars):
            a_lb, a_ub = action_bounds.get_dimension_bounds(action_number)
            if a_lb >= GuardingConstants.GUARD_ACTION:
                at_least_one_guarding = True
                break

        nobody_guarding = False
        if not at_least_one_guarding:
            nobody_guarding = True
            for action_number, action in enumerate(joint_action_vars):
                a_lb, a_ub = action_bounds.get_dimension_bounds(action_number)
                if a_lb >= GuardingConstants.GUARD_ACTION:
                    nobody_guarding = False
                    break


        # Computing bounds for different variables
        expired_lb = rest_lb = guard_lb = 0
        expired_ub = rest_ub = guard_ub = 1
        next_health_lb, next_health_ub = 0, GuardingConstants.MAX_HEALTH_POINTS

        if action_ub <= GuardingConstants.EXPIRED_ACTION:
            expired_lb = 1
            rest_ub = 0
            guard_ub = 0
            next_health_lb = next_health_ub = GuardingConstants.EXPIRED_HEALTH_POINTS
        elif action_lb >= GuardingConstants.REST_ACTION and action_ub <= GuardingConstants.REST_ACTION:
            expired_ub = 0
            rest_lb = 1
            guard_ub = 0

            if at_least_one_guarding:
                next_health_lb = min(health_lb + GuardingConstants.RESTING_REWARD, GuardingConstants.MAX_HEALTH_POINTS)
                next_health_ub = min(health_ub + GuardingConstants.RESTING_REWARD, GuardingConstants.MAX_HEALTH_POINTS)
            elif nobody_guarding:
                next_health_lb = max(health_lb + GuardingConstants.UNGUARDED_REWARD, GuardingConstants.EXPIRED_HEALTH_POINTS)
                next_health_ub = max(health_ub + GuardingConstants.UNGUARDED_REWARD, GuardingConstants.EXPIRED_HEALTH_POINTS)
        elif action_lb >= GuardingConstants.GUARD_ACTION:
            expired_ub = 0
            rest_ub = 0
            guard_lb = 1
            next_health_lb = max(health_lb + GuardingConstants.GUARDING_REWARD, GuardingConstants.EXPIRED_HEALTH_POINTS)
            next_health_ub = max(health_ub + GuardingConstants.GUARDING_REWARD, GuardingConstants.EXPIRED_HEALTH_POINTS)


        # Detecting if at least one agent guards in MILP
        binary_vars = constrs_manager.create_binary_variables(len(joint_action_vars))
        for action_number, action in enumerate(joint_action_vars):
            constrs_to_add.extend([
                constrs_manager.create_indicator_constraint(
                    binary_vars[action_number], 1,
                    constrs_manager.get_assignment_constraint(action, GuardingConstants.GUARD_ACTION)),
                constrs_manager.create_indicator_constraint(
                    binary_vars[action_number], 0,
                    constrs_manager.get_le_constraint(action, GuardingConstants.REST_ACTION)),
            ])


        # Binary variable for checking whether the action is expired, rest or guard
        [expired, rest, guard] = constrs_manager.create_binary_variables(3, lbs=[expired_lb, rest_lb, guard_lb],
                                                                            ubs=[expired_ub, rest_ub, guard_ub])

        constrs_to_add.extend([
            constrs_manager.create_indicator_constraint(
                expired, 1,
                constrs_manager.get_assignment_constraint(action_var, GuardingConstants.EXPIRED_ACTION)),
            constrs_manager.create_indicator_constraint(
                rest, 1,
                constrs_manager.get_assignment_constraint(action_var, GuardingConstants.REST_ACTION)),
            constrs_manager.create_indicator_constraint(
                guard, 1,
                constrs_manager.get_assignment_constraint(action_var, GuardingConstants.GUARD_ACTION)),
            constrs_manager.get_sum_constraint([expired, rest, guard], 1)
        ])

        [next_health_var] = constrs_manager.create_state_variables(1, lbs=[next_health_lb], ubs=[next_health_ub])
        [next_health_var_raw, next_health_var_max] = constrs_manager.create_state_variables(2)
        [at_least_one_guards, nobody_guards] = constrs_manager.create_binary_variables(2)
        constrs_manager.update()

        constrs_to_add.extend([
            constrs_manager.create_indicator_constraint(
                expired, 1,
                constrs_manager.get_linear_constraint([next_health_var_raw, health_var], [1, -1], 0)),
            constrs_manager.create_indicator_constraint(
                guard, 1,
                constrs_manager.get_linear_constraint([next_health_var_raw, health_var], [1, -1],
                                                      GuardingConstants.GUARDING_REWARD)),
            constrs_manager.create_indicator_constraint(
                rest, 1,
                constrs_manager.get_linear_constraint([next_health_var_raw, health_var, at_least_one_guards, nobody_guards],
                                                      [1, -1, -GuardingConstants.RESTING_REWARD, -GuardingConstants.UNGUARDED_REWARD],
                                                      0)),
            constrs_manager.create_indicator_constraint(
                at_least_one_guards, 1,
                constrs_manager.get_linear_constraint(binary_vars, [1] * len(binary_vars), 1, sense=__ge__)),
            constrs_manager.create_indicator_constraint(
                nobody_guards, 1,
                constrs_manager.get_linear_constraint(binary_vars, [1] * len(binary_vars), 0)),
            constrs_manager.get_linear_constraint([at_least_one_guards, nobody_guards], [1, 1], 1)
        ])

        constrs_to_add.extend(
            # max constraint for next health
            constrs_manager.encode_max_constraint(
                next_health_var_max, next_health_var_raw, GuardingConstants.EXPIRED_HEALTH_POINTS, GuardingConstants.BIG_M_CONSTANT) +
            # min constraint for next health
            constrs_manager.encode_min_constraint(
                next_health_var, next_health_var_max, GuardingConstants.MAX_HEALTH_POINTS, GuardingConstants.BIG_M_CONSTANT))

        return [next_health_var], constrs_to_add

    def get_branching_factor(self):
        return GuardingConstants.AGENT_MAX_ACTIONS

    def get_state_dimensions(self):
        return GuardingConstants.AGENT_STATE_DIMENSIONS

    def get_private_state_dimensions(self):
        return GuardingConstants.PERCEPT_IDX
