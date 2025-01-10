import functools
import itertools

from src.verification.bounds.bounds import HyperRectangleBounds
from src.utils.utils import get_widest_bounds
from src.verification.specencoder.breadth_first_compositional_boolean_milp_encoder import CompositionalBooleanMILPEncoder


class BreadthFirstCompositionalSingleAgentCTLMILPEncoder(CompositionalBooleanMILPEncoder):
    # TODO: Replace all lists with sets.
    def __init__(self, constrs_manager, state_vars, agent, env):
        """
        An immutable visitor implementation for constructing a set of MILPs from a bCTL formula
        using the breadth first approach.

        Requires a lot of memory.

        :param constrs_manager: Constraints manager.
        :param state_vars: The current state of the env.
        :param agent: The neural agent.
        :param env: The non-deterministic environment.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        super(BreadthFirstCompositionalSingleAgentCTLMILPEncoder, self).__init__(constrs_manager, state_vars)
        self.bf = env.get_branching_factor()
        self.agent = agent
        self.env = env
        self.next_vars = self.state_vars

    def visitENextFormula(self, element):

        # Add constraints for agent network.
        root_state_vars = self.state_vars
        action_vars, action_constrs = self.agent.get_constraints_for_action(self.constrs_manager, root_state_vars)
        self.constrs_manager.get_variable_tracker().add_action_variables(action_vars)

        next_state_vars = self.constrs_manager.create_state_variables(len(root_state_vars))
        output_bounds = [(float("inf"), float("-inf")) for _ in range(len(next_state_vars))]  # Widest upper and lower bounds for output vars.
        self.constrs_manager.get_variable_tracker().add_state_variables(next_state_vars)

        constrs = []
        for i in range(self.bf):
            constrs_to_add = []

            # Add and get constraints for transition function.
            output_state_vars, output_state_constrs = \
                self.env.get_constraints_for_transition(i, self.constrs_manager, action_vars, root_state_vars)
            constrs_to_add.extend(output_state_constrs)

            # Compute max possible upper and min possible lower bounds for each output var.
            get_widest_bounds(output_bounds, output_state_vars)

            constrs_to_add.extend([self.constrs_manager.get_equality_constraint(nsv, osv)
                                   for nsv, osv in zip(next_state_vars, output_state_vars)])

            constrs.append(constrs_to_add)

        output_lower, output_upper = zip(*output_bounds)  # Unzip the bounds.
        self.constrs_manager.add_variable_bounds(next_state_vars, HyperRectangleBounds(output_lower, output_upper))

        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ENextFormula
            smaller_formula = ENextFormula(k - 1, element.left)
        self.state_vars = next_state_vars
        left_constraints = smaller_formula.acceptI(self)
        self.state_vars = root_state_vars
        product = [a + l + r for a, l, r in itertools.product([action_constrs], constrs, left_constraints)]
        return product

    def visitANextFormula(self, element):
        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ANextFormula
            smaller_formula = ANextFormula(k - 1, element.left)
        full_transition_milps = []
        root_state_vars = self.state_vars
        for i in range(self.bf):
            # Add constraints for agent network.
            action_grb_vars, action_constrs = self.agent.get_constraints_for_action(self.constrs_manager, root_state_vars)

            # Add and get constraints for transition function.
            output_state_vars, output_state_constrs = \
                self.env.get_constraints_for_transition(i, self.constrs_manager, action_grb_vars, root_state_vars)

            current_transition_milp = [action_constrs + output_state_constrs]

            self.state_vars = output_state_vars

            smaller_formula_milps = smaller_formula.acceptI(self)
            full_transition_milp = list(l + r for l, r in itertools.product(current_transition_milp, smaller_formula_milps))
            full_transition_milps.append(full_transition_milp)
        result = list(functools.reduce(lambda x, y: x + y, l) for l in itertools.product(*full_transition_milps))
        return result
