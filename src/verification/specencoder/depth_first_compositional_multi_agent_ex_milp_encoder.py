import math

from src.utils.formula_visitors.bounds_satisfaction_visitor import BoundsBooleanFormulaSatisfactionVisitor
from src.utils.formula_visitors.immutable_formula_visitor_interface import FormulaVisitorI
from src.utils.utils import increment_indices
from src.verification.bounds.bounds import HyperRectangleBounds


class DepthFirstCompositionalMultiAgentExistsNextMILPEncoder(FormulaVisitorI):
    """
    This encoder only support EX, disjunction and conjunction of atomic formulas.
    """
    def __init__(self, constrs_manager, state_vars, agents, env):
        """
        An immutable visitor implementation for constructing a set of MILPs from a (EX,AND,OR) formula.
        :param constrs_manager: Constraints manager.
        :param state_vars: The current state of the env.
        :param agents: The agents.
        :param env: The environment.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        self.state_vars = state_vars
        self.constrs_manager = constrs_manager
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


        # job_count to keep track of the number of created jobs
        self.job_count = 1
        # simulates the stack to keep track of the set of constraints
        # when computing MILPs in the depth first fashion
        self.constrs_stack = []
        # the splitting process that will add the MILPs to the jobs queue
        self.splitting_process = None

    def set_splitting_process(self, splitting_process):
        self.splitting_process = splitting_process

    def visitConstraintFormula(self, element):
        state_bounds = self.constrs_manager.get_variable_bounds(self.state_vars)

        # Only add the MILP if the constraint does not clash with
        # the state bounds. Otherwise, the MILP will be trivially infeasible
        if element.acceptI(BoundsBooleanFormulaSatisfactionVisitor(state_bounds)):
            constrs_to_add = [self.constrs_manager.get_atomic_constraint(element, self.state_vars)]

            self.constrs_stack.append(constrs_to_add)
            self.splitting_process.add_job(self.job_count, self.constrs_stack,
                                           self.constrs_manager.get_variable_tracker().get_trace())
            self.job_count += 1
            self.constrs_stack.pop()

        return self.job_count

    def visitVarVarConstraintFormula(self, element):
        return self.visitConstraintFormula(element)

    def visitVarConstConstraintFormula(self, element):
        return self.visitConstraintFormula(element)

    def visitDisjFormula(self, element):
        element.left.acceptI(self)
        element.right.acceptI(self)

        return self.job_count

    def visitNAryDisjFormula(self, element):
        state_bounds = self.constrs_manager.get_variable_bounds(self.state_vars)

        # Only add the MILP if the constraint does not clash with
        # the state bounds. Otherwise, the MILP will be trivially infeasible
        if element.acceptI(BoundsBooleanFormulaSatisfactionVisitor(state_bounds)):
            deltas = self.constrs_manager.create_binary_variables(len(element.clauses))
            constrs_to_add = []
            for i, clause in enumerate(element.clauses):

                constrs_to_add.append(self.constrs_manager.create_indicator_constraint(
                    deltas[i], 1, clause.get_custom_atomic_constraint(self.state_vars)))

            constrs_to_add.append(self.constrs_manager.get_sum_constraint(deltas, 1))

            self.constrs_stack.append(constrs_to_add)
            self.splitting_process.add_job(self.job_count, self.constrs_stack,
                                           self.constrs_manager.get_variable_tracker().get_trace())
            self.job_count += 1
            self.constrs_stack.pop()

        return self.job_count

    def visitAtomicDisjFormula(self, element):
        state_bounds = self.constrs_manager.get_variable_bounds(self.state_vars)

        # Only add the MILP if the constraint does not clash with
        # the state bounds. Otherwise, the MILP will be trivially infeasible
        if element.acceptI(BoundsBooleanFormulaSatisfactionVisitor(state_bounds)):
            left_constr = element.left.get_custom_atomic_constraint(self.state_vars)
            right_constr = element.right.get_custom_atomic_constraint(self.state_vars)
            constrs_to_add = []

            deltas = self.constrs_manager.create_binary_variables(2)
            constrs_to_add.append(self.constrs_manager.create_indicator_constraint(deltas[0], 1, left_constr))
            constrs_to_add.append(self.constrs_manager.create_indicator_constraint(deltas[1], 1, right_constr))
            constrs_to_add.append(self.constrs_manager.get_sum_constraint(deltas, 1))

            self.constrs_stack.append(constrs_to_add)
            self.splitting_process.add_job(self.job_count, self.constrs_stack,
                                           self.constrs_manager.get_variable_tracker().get_trace())
            self.job_count += 1
            self.constrs_stack.pop()

        return self.job_count

    def visitAtomicConjFormula(self, element):
        state_bounds = self.constrs_manager.get_variable_bounds(self.state_vars)

        # Only add the MILP if the constraint does not clash with
        # the state bounds. Otherwise, the MILP will be trivially infeasible
        if element.acceptI(BoundsBooleanFormulaSatisfactionVisitor(state_bounds)):
            left_constr = element.left.get_custom_atomic_constraint(self.state_vars)
            right_constr = element.right.get_custom_atomic_constraint(self.state_vars)
            # left_constrs and right_constrs already have the same root variables
            constrs_to_add = [left_constr, right_constr]

            self.constrs_stack.append(constrs_to_add)
            self.splitting_process.add_job(self.job_count, self.constrs_stack,
                                           self.constrs_manager.get_variable_tracker().get_trace())
            self.job_count += 1
            self.constrs_stack.pop()

        return self.job_count

    def visitConjFormula(self, element):
        raise Exception("Arbitrary conjunction is not supported")

    def visitNAryConjFormula(self, element):
        """
        Here we assume that element.clauses are atomic formulas
        :param element:
        :return:
        """
        state_bounds = self.constrs_manager.get_variable_bounds(self.state_vars)

        print("Reached a leaf")

        # Only add the MILP if the constraint does not clash with
        # the state bounds. Otherwise, the MILP will be trivially infeasible
        if element.acceptI(BoundsBooleanFormulaSatisfactionVisitor(state_bounds)):
            print("Creating a job")
            constrs_to_add = [clause.get_custom_atomic_constraint(self.state_vars) for clause in element.clauses]

            self.constrs_stack.append(constrs_to_add)
            self.splitting_process.add_job(self.job_count, self.constrs_stack,
                                           self.constrs_manager.get_variable_tracker().get_trace())
            self.job_count += 1
            self.constrs_stack.pop()

        return self.job_count

    def visitENextFormula(self, element):
        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ENextFormula
            smaller_formula = ENextFormula(k - 1, element.left)

        # Add constraints for agent network.
        init_vars = self.state_vars

        #############################################
        ######### The observation functions #########
        #############################################
        agent_perception_vars = []
        agent_perception_constrs = []
        for agent_number, agent in enumerate(self.agents):
            perception_vars, observation_constrs = \
                agent.get_constraints_for_obs(self.constrs_manager,
                                              self.get_local_state_vars(init_vars, agent_number))
            agent_perception_constrs.extend(observation_constrs)
            agent_perception_vars.append(perception_vars)

        self.constrs_stack.append(agent_perception_constrs)

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

        action_ranges = [agent.get_branching_factor() for agent in self.agents] + \
                        [self.env.get_branching_factor()]
        action_indices = [0 for _ in range(len(action_ranges))]
        global_action_idx = 0

        for i in range(self.num_joint_actions):
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
            next_state_vars = []
            for agent_number in range(len(self.agents)):
                next_state_vars.extend(agent_output_state_vars[agent_number])
                next_state_vars.extend(agent_perception_vars[agent_number])
            next_state_vars.extend(env_output_state_vars)

            next_state_bounds = [(var.lb, var.ub) for var in next_state_vars]
            next_state_lower, next_state_upper = zip(*next_state_bounds)  # Unzip the bounds.
            self.constrs_manager.add_variable_bounds(next_state_vars, HyperRectangleBounds(next_state_lower, next_state_upper))

            self.constrs_stack.append(transition_constrs + all_action_constrs)
            self.constrs_manager.get_variable_tracker().add_state_variables(next_state_vars)
            self.constrs_manager.get_variable_tracker().add_action_variables(all_actions_vars)

            # The recursive call
            self.state_vars = next_state_vars
            smaller_formula.acceptI(self)

            # Pop the constraints from this branch
            self.constrs_stack.pop()
            self.constrs_manager.get_variable_tracker().pop_state_variables()
            self.constrs_manager.get_variable_tracker().pop_action_variables()

            # Increment global action index and the action indices
            global_action_idx += 1
            stop = increment_indices(action_indices, action_ranges)
            if stop:
                break


        # Undo the changes done in this method
        self.constrs_stack.pop()
        self.state_vars = init_vars

        return self.job_count

    def visitANextFormula(self, element):
        raise Exception("AX is not supported")

    def get_local_state_vars(self, init_vars, agent_number):
        if agent_number < len(self.local_state_offsets) - 1:
            return init_vars[self.local_state_offsets[agent_number]:self.local_state_offsets[agent_number+1]]
        else:
            return init_vars[self.local_state_offsets[agent_number]:]
