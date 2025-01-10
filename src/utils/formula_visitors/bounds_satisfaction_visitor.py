from src.utils.formula import *
from src.utils.formula_visitors.immutable_formula_visitor_interface import FormulaVisitorI

INVERTED_SENSE = {LE: GT, GT: LE, GE: LT, LT: GE, EQ: NE, NE: EQ}


class BoundsBooleanFormulaSatisfactionVisitor(FormulaVisitorI):
    """
    An immutable visitor implementation for checking whether given lower and upper bounds
    do not contradict a formula.

    This visitor is used to understand when given bounds clash with the formula,
    leading to trivial infeasibility.
    """

    def __init__(self, bounds):
        self.lower_bounds = bounds.get_lower()
        self.upper_bounds = bounds.get_upper()

    def visitVarConstConstraintFormula(self, element):
        sense = element.sense
        if sense == LE:
            return self.lower_bounds[element.op1.i] <= element.op2
        elif sense == GE:
            return self.upper_bounds[element.op1.i] >= element.op2
        raise ValueError("Expected non strict inequality. Instead got", sense)

    def visitVarVarConstraintFormula(self, element):
        sense = element.sense
        if sense == LE:
            return self.lower_bounds[element.op1.i] <= self.upper_bounds[element.op2.i]
        elif sense == GE:
            return self.upper_bounds[element.op1.i] >= self.lower_bounds[element.op2.i]
        raise ValueError("Expected non strict inequality. Instead got", sense)

    def visitConjFormula(self, element):
        return element.left.acceptI(self) and element.right.acceptI(self)

    def visitAtomicConjFormula(self, element):
        return element.left.acceptI(self) and element.right.acceptI(self)

    def visitNAryConjFormula(self, element):
        for e in element.clauses:
            if not e.acceptI(self):
                return False
        return True

    def visitDisjFormula(self, element):
        return element.left.acceptI(self) or element.right.acceptI(self)

    def visitAtomicDisjFormula(self, element):
        return element.left.acceptI(self) or element.right.acceptI(self)

    def visitNAryDisjFormula(self, element):
        for e in element.clauses:
            if e.acceptI(self):
                return True
        return False
