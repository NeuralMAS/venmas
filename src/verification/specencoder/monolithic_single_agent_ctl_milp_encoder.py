from src.utils.utils import get_widest_bounds
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.specencoder.monolithic_boolean_milp_encoder import MonolithicBooleanMILPEncoder


class MonolithicSingleAgentCTLMILPEncoder(MonolithicBooleanMILPEncoder):
    def __init__(self, constrs_manager, state_vars, agent, env):
        """
        An immutable visitor implementation for constructing a single MILP from a CTL formula
        for a single agent-environment system.

        Here the assumption is
           * there is a single agent
           * the agent has a protocol function that returns the action for a given input state
           * the environment has a non-deterministic transition function, i.e., it returns the set of possible next states

        :param constrs_manager: Constraints manager.
        :param state_vars: The current state of the env.
        :param agent: The neural agent.
        :param env: The non-deterministic environment.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        super(MonolithicSingleAgentCTLMILPEncoder, self).__init__(constrs_manager, state_vars)
        self.bf = env.get_branching_factor()
        self.agent = agent
        self.env = env
        self.next_vars = self.state_vars

    def visitENextFormula(self, element):
        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ENextFormula
            smaller_formula = ENextFormula(k - 1, element.left)

        constrs_to_add = []

        root_state_vars = self.state_vars

        # Add constraints for agent network.
        action_grb_vars, action_constrs = self.agent.get_constraints_for_action(self.constrs_manager, root_state_vars)
        constrs_to_add.extend(action_constrs)
        self.constrs_manager.get_variable_tracker().add_action_variables(action_grb_vars)

        d = self.constrs_manager.create_binary_variables(self.bf)
        constrs_to_add.append(self.constrs_manager.get_sum_constraint(d, 1))

        next_state_vars = self.constrs_manager.create_state_variables(len(self.state_vars))
        output_bounds = [(float("inf"), float("-inf")) for _ in range(len(next_state_vars))]  # Widest upper and lower bounds for output vars.
        self.constrs_manager.get_variable_tracker().add_state_variables(next_state_vars)

        self.constrs_manager.update()
        for i in range(self.bf):

            # Add and get constraints for transition function.
            output_state_vars, output_state_constrs = \
                self.env.get_constraints_for_transition(i, self.constrs_manager, action_grb_vars, root_state_vars)

            # Compute max possible upper and min possible lower bounds for each output var.
            get_widest_bounds(output_bounds, output_state_vars)

            next_var_constrs = [nsv == osv for nsv, osv in zip(next_state_vars, output_state_vars)]

            for constr in (output_state_constrs + next_var_constrs):
                if constr._sense != 'I':  # Check if indicator constraint.
                    constrs_to_add.append(self.constrs_manager.create_indicator_constraint(d[i], 1, constr))
                else:
                    constrs_to_add.append(constr)

        output_lower, output_upper = zip(*output_bounds)  # Unzip the bounds.
        self.constrs_manager.add_variable_bounds(next_state_vars, HyperRectangleBounds(output_lower, output_upper))

        self.state_vars = next_state_vars
        left_constraints = smaller_formula.acceptI(self)
        constrs_to_add.extend(left_constraints)
        self.state_vars = root_state_vars

        return constrs_to_add

    def visitANextFormula(self, element):
        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ANextFormula
            smaller_formula = ANextFormula(k - 1, element.left)

        constrs_to_add = []
        root_state_vars = self.state_vars

        for i in range(self.bf):
            # Add constraints for agent network.
            action_grb_vars, action_constrs = self.agent.get_constraints_for_action(self.constrs_manager, root_state_vars)
            constrs_to_add.extend(action_constrs)

            # Add and get constraints for transition function.
            output_state_vars, output_state_constrs = \
                self.env.get_constraints_for_transition(i, self.constrs_manager, action_grb_vars, root_state_vars)

            constrs_to_add.extend(output_state_constrs)
            self.state_vars = output_state_vars
            constrs_to_add.extend(smaller_formula.acceptI(self))

        return constrs_to_add

