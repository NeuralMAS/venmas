from src.utils.formula_visitors.immutable_formula_visitor_interface import FormulaVisitorI
from src.utils.utils import get_widest_bounds
from src.verification.bounds.bounds import HyperRectangleBounds


class FormulaRNNMILPBuilderVisitor(FormulaVisitorI):
    def __init__(self, constrs_manager, state_vars, agent, env):
        """
        An immutable visitor implementation for constructing a single MILP from a formula.
        :param constrs_manager: Constraints manager.
        :param state_vars: The current state of the env.
        :param agent: The neural agent.
        :param env: The non-deterministic environment.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        self.state_vars = [state_vars]
        self.constrs_manager = constrs_manager
        self.bf = env.get_branching_factor()
        print("Branching factor", self.bf)
        self.agent = agent
        self.env = env
        self.next_vars = self.state_vars
        self.uid = 0

    def visitConstraintFormula(self, element):
        constrs_to_add = [self.constrs_manager.get_atomic_constraint(element, self.state_vars[-1])]
        return constrs_to_add

    def visitVarVarConstraintFormula(self, element):
        return self.visitConstraintFormula(element)

    def visitVarConstConstraintFormula(self, element):
        return self.visitConstraintFormula(element)

    def visitDisjFormula(self, element):
        [d] = self.constrs_manager.create_binary_variables(1)
        self.constrs_manager.update()

        binvars = set()
        constrs_to_add = []
        init_vars = self.state_vars[:]

        next_vars_x1 = self.constrs_manager.create_state_variables(len(self.state_vars))
        x1_constrs = [vi == v for v, vi in zip(next_vars_x1, init_vars)]
        self.state_vars = next_vars_x1
        left_x1_constrs = element.left.acceptI(self)
        self.constrs_manager.update()

        for constr in (x1_constrs + left_x1_constrs):
            if constr._sense != 'I':  # Hack to check if indicator constraint.
                constrs_to_add.append(self.constrs_manager.create_indicator_constraint(binvars, d, 1, constr))
            else:
                constrs_to_add.append(constr)

        next_vars_x2 = self.constrs_manager.create_state_variables(len(self.state_vars))
        x2_constrs = [vi == v for v, vi in zip(next_vars_x2, init_vars)]
        self.state_vars = next_vars_x2
        right_x2_constrs = element.right.acceptI(self)

        for constr in (x2_constrs + right_x2_constrs):
            if constr._sense != 'I':  # Check if indicator constraint.
                constrs_to_add.append(self.constrs_manager.create_indicator_constraint(binvars, d, 0, constr))
            else:
                constrs_to_add.append(constr)

        self.constrs_manager.binvars.update(binvars)
        return constrs_to_add

    def visitConjFormula(self, element):
        left_constraints = element.left.acceptI(self)
        right_constraints = element.right.acceptI(self)
        constrs_to_add = left_constraints + right_constraints
        return constrs_to_add

    def visitENextFormula(self, element):
        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ENextFormula
            smaller_formula = ENextFormula(k - 1, element.left)

        binvars = set()
        constrs_to_add = []

        flat_state_vars = [item for sublist in self.state_vars for item in sublist]
        current_state_vars = self.state_vars[-1]

        # Add constraints for agent network.
        action_grb_vars, action_constrs, abvs = \
            self.agent.get_constraints_for_action(self.constrs_manager, flat_state_vars, len(self.state_vars))
        binvars.update(abvs)
        constrs_to_add.extend(action_constrs)
        self.constrs_manager.get_variable_tracker().add_action_variables(action_grb_vars)

        d = self.constrs_manager.create_binary_variables(self.bf)
        constrs_to_add.append(self.constrs_manager.get_sum_constraint(d, 1))

        next_state_vars = self.constrs_manager.create_state_variables(len(current_state_vars))
        output_bounds = [(float("inf"), float("-inf")) for _ in range(len(next_state_vars))]  # Widest upper and lower bounds for output vars.
        self.constrs_manager.get_variable_tracker().add_state_variables(next_state_vars)

        self.constrs_manager.update()
        for i in range(self.bf):

            # Add and get constraints for transition function.
            output_state_vars, output_state_constrs, obvs = \
                self.env.get_constraints_for_transition(i, self.constrs_manager, action_grb_vars, current_state_vars)
            binvars.update(obvs)

            # Compute max possible upper and min possible lower bounds for each output var.
            get_widest_bounds(output_bounds, output_state_vars)

            next_var_constrs = [nsv == osv for nsv, osv in zip(next_state_vars, output_state_vars)]

            for constr in (output_state_constrs + next_var_constrs):
                if constr._sense != 'I':  # Check if indicator constraint.
                    constrs_to_add.append(self.constrs_manager.create_indicator_constraint(binvars, d[i], 1, constr))
                else:
                    constrs_to_add.append(constr)

        output_lower, output_upper = zip(*output_bounds)  # Unzip the bounds.
        self.constrs_manager.add_variable_bounds(next_state_vars, HyperRectangleBounds(output_lower, output_upper))

        self.state_vars = self.state_vars + [next_state_vars]
        left_constraints = smaller_formula.acceptI(self)
        constrs_to_add.extend(left_constraints)

        return constrs_to_add

    def visitANextFormula(self, element):
        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ANextFormula
            smaller_formula = ANextFormula(k - 1, element.left)

        constrs_to_add = []

        flat_state_vars = [item for sublist in self.state_vars for item in sublist]
        current_state_vars = self.state_vars[-1]
        original_state_vars = self.state_vars

        for i in range(self.bf):
            # Add constraints for agent network.
            action_grb_vars, action_constrs, _ = \
                self.agent.get_constraints_for_action(self.constrs_manager, flat_state_vars, len(self.state_vars))
            constrs_to_add.extend(action_constrs)

            # Add and get constraints for transition function.
            output_state_vars, output_state_constrs, _ = \
                self.env.get_constraints_for_transition(i, self.constrs_manager, action_grb_vars, current_state_vars)

            constrs_to_add.extend(output_state_constrs)
            self.state_vars = original_state_vars + [output_state_vars]
            constrs_to_add.extend(smaller_formula.acceptI(self))
        return constrs_to_add

