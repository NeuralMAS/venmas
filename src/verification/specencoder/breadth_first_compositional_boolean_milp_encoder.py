import itertools

from src.utils.formula_visitors.immutable_formula_visitor_interface import FormulaVisitorI


class CompositionalBooleanMILPEncoder(FormulaVisitorI):
    def __init__(self, constrs_manager, state_vars):
        """
        A compositional implementation for the Boolean fragment of formula.
        Handles atomic formulas and conjunction and disjunction of formulas.

        :param constrs_manager: Constraints manager.
        :param state_vars: The current state of the env.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        self.state_vars = state_vars
        self.constrs_manager = constrs_manager

    def visitConstraintFormula(self, element):
        constrs = [self.constrs_manager.get_atomic_constraint(element, self.state_vars)]
        return [constrs]

    def visitVarVarConstraintFormula(self, element):
        constrs = self.visitConstraintFormula(element)
        return constrs

    def visitVarConstConstraintFormula(self, element):
        constrs = self.visitConstraintFormula(element)
        return constrs

    def visitDisjFormula(self, element):
        left_constrs = element.left.acceptI(self)
        right_constrs = element.right.acceptI(self)
        # left_constrs and right_constrs already have the same root variables
        constrs = left_constrs + right_constrs
        return constrs

    def visitAtomicDisjFormula(self, element):
        left_constr = element.left.get_custom_atomic_constraint(self.state_vars)
        right_constr = element.right.get_custom_atomic_constraint(self.state_vars)
        constrs = []
        # left_constrs and right_constrs already have the same root variables
        deltas = self.constrs_manager.create_binary_variables(2)
        constrs.append(self.constrs_manager.create_indicator_constraint(deltas[0], 1, left_constr))
        constrs.append(self.constrs_manager.create_indicator_constraint(deltas[1], 1, right_constr))
        constrs.append(self.constrs_manager.get_sum_constraint(deltas, 1))
        return [constrs]

    def visitAtomicConjFormula(self, element):
        left_constr = element.left.get_custom_atomic_constraint(self.state_vars)
        right_constr = element.right.get_custom_atomic_constraint(self.state_vars)
        # left_constrs and right_constrs already have the same root variables
        constrs = [left_constr, right_constr]
        return [constrs]

    def visitConjFormula(self, element):
        left_constrs = element.left.acceptI(self)
        right_constrs = element.right.acceptI(self)
        # left_constrs and right_constrs already have the same root variables
        constrs = [l + r for l, r in itertools.product(left_constrs, right_constrs)]
        return constrs
