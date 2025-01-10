from src.utils.formula_visitors.bounds_satisfaction_visitor import BoundsBooleanFormulaSatisfactionVisitor
from src.utils.formula_visitors.immutable_formula_visitor_interface import FormulaVisitorI


class DepthFirstCompositionalSingleAgentExistsNextMILPEncoder(FormulaVisitorI):
    """
    This encoder only support EX, disjunction and conjunction of atomic formulas.
    """
    def __init__(self, constrs_manager, state_vars, agent, env):
        """
        An immutable visitor implementation for constructing a set of MILPs from a (EX,AND,OR) formula.

        :param constrs_manager: Constraints manager.
        :param state_vars: The current state of the env.
        :param bf: The branching factor of the transition function
        :param agent: The neural agent.
        :param env: The non-deterministic environment.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        self.state_vars = state_vars
        self.constrs_manager = constrs_manager
        self.bf = env.get_branching_factor()
        self.agent = agent
        self.env = env
        self.next_vars = self.state_vars

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

    def visitENextFormula(self, element):
        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ENextFormula
            smaller_formula = ENextFormula(k - 1, element.left)

        # Add constraints for agent network.
        root_state_vars = self.state_vars

        action_vars, action_constrs = self.agent.get_constraints_for_action(self.constrs_manager, root_state_vars)
        self.constrs_stack.append(action_constrs)
        self.constrs_manager.get_variable_tracker().add_action_variables(action_vars)

        for i in range(self.env.get_branching_factor_opt(root_state_vars, action_vars)):
            # Add and get constraints for transition function.
            next_state_vars, output_state_constrs = \
                self.env.get_constraints_for_transition(i, self.constrs_manager, action_vars, root_state_vars)

            self.constrs_stack.append(output_state_constrs)
            self.constrs_manager.get_variable_tracker().add_state_variables(next_state_vars)

            # The recursive call
            self.state_vars = next_state_vars
            smaller_formula.acceptI(self)

            # Pop the constraints from this branch
            self.constrs_stack.pop()
            self.constrs_manager.get_variable_tracker().pop_state_variables()

        # Undo the changes done in this method
        self.constrs_stack.pop()
        self.constrs_manager.get_variable_tracker().pop_action_variables()
        self.state_vars = root_state_vars

        return self.job_count

    def visitANextFormula(self, element):
        raise Exception("AX is not supported")
