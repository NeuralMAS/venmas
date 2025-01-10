from src.utils.constants import NOT_IMPLEMENTED
from src.utils.formula_visitors.immutable_formula_visitor_interface import FormulaVisitorI


class MonolithicBooleanMILPEncoder(FormulaVisitorI):
    def __init__(self, constrs_manager, state_vars):
        """
        A monolithic implementation for the Boolean fragment of formula.
        Handles atomic formulas and conjunction and disjunction of formulas.

        Supposed to be extended for CTL or ATL specific encodings.

        :param constrs_manager: Constraints manager (custom or gurobi).
        :param state_vars: The variables referring o the current state of the env.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        self.state_vars = state_vars
        self.constrs_manager = constrs_manager

    def visitConstraintFormula(self, element):
        constrs_to_add = [self.constrs_manager.get_atomic_constraint(element, self.state_vars)]
        return constrs_to_add

    def visitVarVarConstraintFormula(self, element):
        return self.visitConstraintFormula(element)

    def visitVarConstConstraintFormula(self, element):
        return self.visitConstraintFormula(element)

    def visitNAryDisjFormula(self, element):

        split_vars = self.constrs_manager.create_binary_variables(len(element.clauses))
        self.constrs_manager.update()

        constrs = []
        for i in range(len(element.clauses)):
            current_clause_constrs = element.clauses[i].acceptI(self)

            for disj_constr in current_clause_constrs:
                if disj_constr._sense != 'I':  # Hack to check if indicator constraint.
                    constrs.append(self.constrs_manager.create_indicator_constraint(split_vars[i], 1, disj_constr))
                else:
                    constrs.append(disj_constr)

        # exactly one variable must be true
        constrs.append(self.constrs_manager.get_sum_constraint(split_vars, 1))

        return constrs

    def visitDisjFormula(self, element):
        [d1, d2] = self.constrs_manager.create_binary_variables(2)
        self.constrs_manager.update()

        # exactly one variable must be true
        constrs_to_add = [self.constrs_manager.get_sum_constraint([d1, d2], 1)]

        left_x1_constrs = element.left.acceptI(self)
        right_x2_constrs = element.right.acceptI(self)
        for side, d in [(left_x1_constrs, d1), (right_x2_constrs, d2)]:
            for constr in side:
                if constr._sense != 'I':  # Check if indicator constraint.
                    constrs_to_add.append(self.constrs_manager.create_indicator_constraint(d, 1, constr))
                else:
                    constrs_to_add.append(constr)

        return constrs_to_add

    def visitAtomicDisjFormula(self, element):
        return self.visitDisjFormula(element)

    def visitAtomicConjFormula(self, element):
        return self.visitConjFormula(element)

    def visitConjFormula(self, element):
        left_constraints = element.left.acceptI(self)
        right_constraints = element.right.acceptI(self)
        constrs_to_add = left_constraints + right_constraints
        return constrs_to_add

    def visitNAryConjFormula(self, element):

        constrs = []
        for clause in element.clauses:
            constrs.extend(clause.acceptI(self))

        return constrs

    def visitENextFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    def visitANextFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)
