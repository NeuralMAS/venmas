import math

from src.utils.utils import get_widest_bounds, increment_indices
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.specencoder.monolithic_boolean_milp_encoder import MonolithicBooleanMILPEncoder


class MonolithicMultiAgentCTLMILPEncoder(MonolithicBooleanMILPEncoder):
    def __init__(self, constrs_manager, state_vars, agents, env):
        """
        An immutable visitor implementation for constructing a single MILP from a EX^k formula
        for a NIS as defined in Akintunde et al, KR 2020.

        Here the assumption is
           * every agent has an observation function (implemented by a NN)
           * every agent has a symbolic protocol function that returns the set of available actions
           * every agent has its local transition function
           * the environment has its protocol function
           * the environment has its local transition function

        :param constrs_manager: Constraints manager.
        :param state_vars: The global state of the system, includes the local states of the agents and the environment.
        :param agents: The set of agents.
        :param env: The environment.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        super(MonolithicMultiAgentCTLMILPEncoder, self).__init__(constrs_manager, state_vars)
        self.env = env
        self.next_vars = self.state_vars

        # all agents, potentially including zero-one agent
        self.agents = agents
        self.agent_count = len(agents)

        # The total number of all possible actions
        self.num_joint_actions = math.prod([agent.get_branching_factor() for agent in self.agents]) * \
                                 env.get_branching_factor()

        # The offsets for figuring out the portion of the global state corresponding to each agent and the environment
        self.local_state_offsets = []
        offset = 0
        for agent_number in range(self.agent_count):
            self.local_state_offsets.append(offset)
            offset += agents[agent_number].get_state_dimensions()
        # the last one is the offset for the local environment state
        self.local_state_offsets.append(offset)

    def visitENextFormula(self, element):
        """
        Existential formula visitor.
        :param element:
        :return:
        """

        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ENextFormula
            smaller_formula = ENextFormula(k - 1, element.left)

        constrs_to_add = []

        # Allocate binary variables for each possible combination of actions.
        delta = self.constrs_manager.create_binary_variables(self.num_joint_actions)
        constrs_to_add.append(self.constrs_manager.get_sum_constraint(delta, 1))

        self.constrs_manager.update()

        init_vars = self.state_vars

        #############################################
        ######### The observation functions #########
        #############################################
        agent_perception_vars = []
        for agent_number, agent in enumerate(self.agents):
            perception_vars, observation_constrs = \
                agent.get_constraints_for_obs(self.constrs_manager,
                                              self.get_local_state_vars(init_vars, agent_number))
            constrs_to_add.extend(observation_constrs)
            agent_perception_vars.append(perception_vars)


        ##########################################
        ######### The protocol functions #########
        ##########################################

        ### the protocol functions for all agents
        # As an optimisation we assume here that the protocol function returns a list of action vars and a list of
        # action constraints, one for each possible action (as per the branching factor)
        action_vars = []
        action_constrs = []
        for agent_number, agent in enumerate(self.agents):
            local_state = self.get_local_state_vars(init_vars, agent_number)
            local_state_with_new_perception = (local_state[:agent.get_private_state_dimensions()] +
                                               agent_perception_vars[agent_number])
            agent_action_vars, agent_action_constrs = \
                agent.get_constraints_for_protocol(self.constrs_manager, local_state_with_new_perception)

            action_vars.append(agent_action_vars)
            action_constrs.append(agent_action_constrs)

        ### the protocol functions for the environment
        env_action_vars, env_action_constrs = \
            self.env.get_constraints_for_protocol(self.constrs_manager,
                                                  self.get_local_state_vars(init_vars, self.agent_count))

        #####################################################################
        ######### The transition functions and the actual branching #########
        #####################################################################

        # next_state_vars are the variables for the next state.
        next_state_vars = self.constrs_manager.create_state_variables(len(self.state_vars))
        next_state_bounds = [(float("inf"), float("-inf")) for _ in range(len(next_state_vars))]  # Widest upper and lower bounds for output vars.
        self.constrs_manager.get_variable_tracker().add_state_variables(next_state_vars)

        # action vars for tracking and printing traces
        joint_actions_count = sum([len(action_vars[agent_number][0]) for agent_number in range(self.agent_count)])
        next_action_vars = self.constrs_manager.create_state_variables(joint_actions_count)
        self.constrs_manager.get_variable_tracker().add_action_variables(next_action_vars)

        # indices to keep track of the current combination of actions
        # We iterate through all possible indices, each bound by the corresponding branching factor
        action_ranges = [agent.get_branching_factor() for agent in self.agents] + \
                        [self.env.get_branching_factor()]
        action_indices = [0 for _ in range(len(action_ranges))]
        global_action_idx = 0

        while True:
            ### Get all action vars and action constras according to the current indices
            # Flattened for convenience already here
            all_actions_vars = [item for row in [action_vars[agent_number][action_indices[agent_number]]
                                for agent_number in range(self.agent_count)] for item in row]
            all_action_constrs = [item for row in [action_constrs[agent_number][action_indices[agent_number]]
                                  for agent_number in range(self.agent_count)] for item in row]

            ### the transition functions for all template agents
            agent_output_state_vars = []
            transition_constrs = []
            for agent_number, agent in enumerate(self.agents):
                agent_next_local_state_vars, agent_trans_constrs = \
                    agent.get_constraints_for_transition(self.constrs_manager,
                                                         self.get_local_state_vars(init_vars, agent_number),
                                                         action_vars[agent_number][action_indices[agent_number]],
                                                         all_actions_vars, env_action_vars[action_indices[-1]])

                agent_output_state_vars.append(agent_next_local_state_vars)
                transition_constrs.extend(agent_trans_constrs)

            ### the transition function for the environment
            env_output_state_vars, env_trans_constrs = \
                self.env.get_constraints_for_transition(self.constrs_manager,
                                                        self.get_local_state_vars(init_vars, self.agent_count),
                                                        env_action_vars[action_indices[-1]], all_actions_vars)
            transition_constrs.extend(env_trans_constrs)

            ### Put together the variables of the next global state from the agents local states and their perceptions
            output_state_vars = []
            for agent_number in range(len(self.agents)):
                output_state_vars.extend(agent_output_state_vars[agent_number])
                output_state_vars.extend(agent_perception_vars[agent_number])
            output_state_vars.extend(env_output_state_vars)

            # Compute max possible upper and min possible lower bounds for each output var.
            get_widest_bounds(next_state_bounds, output_state_vars)

            # Constraints for identifying the next_state_vars with the output_state_vars
            # for the current action combination
            next_var_constrs = [nsv == osv for nsv, osv in zip(next_state_vars, output_state_vars)]
            # Constraints for identifying the next_action_vars with the current all_action_vars
            next_action_constrs = [nsv == osv for nsv, osv in zip(next_action_vars, all_actions_vars)]

            for constr in all_action_constrs + transition_constrs + next_var_constrs + next_action_constrs:
                if constr._sense != 'I':  # Hack to check if indicator constraint.
                    constrs_to_add.append(self.constrs_manager
                                          .create_indicator_constraint(delta[global_action_idx], 1, constr))
                else:
                    constrs_to_add.append(constr)

            # Increment global action index and the action indices
            global_action_idx += 1
            stop = increment_indices(action_indices, action_ranges)
            if stop:
                break

        output_lower, output_upper = zip(*next_state_bounds)  # Unzip the bounds.
        self.constrs_manager.add_variable_bounds(next_state_vars, HyperRectangleBounds(output_lower, output_upper))

        self.state_vars = next_state_vars

        left_constraints = smaller_formula.acceptI(self)
        constrs_to_add.extend(left_constraints)

        self.state_vars = init_vars

        return constrs_to_add

    def get_local_state_vars(self, init_vars, agent_number):
        if agent_number < len(self.local_state_offsets) - 1:
            return init_vars[self.local_state_offsets[agent_number]:self.local_state_offsets[agent_number+1]]
        else:
            return init_vars[self.local_state_offsets[agent_number]:]
