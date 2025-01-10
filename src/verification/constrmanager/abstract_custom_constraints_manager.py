from gurobipy import *

from src.utils.constraint import WeightedSum, LinearConstraint, IndicatorConstraint, MaxConstraint, MinConstraint
from src.utils.variable import ContinuousVariable, BinaryVariable, IntegerVariable
from src.verification.constrmanager.aes_variable_tracker import AESVariableTracker
from src.verification.constrmanager.constraints_manager_interface import ConstraintsManagerI
from src.verification.constrmanager.execution_statistics import ExecutionStatistics
from operator import __le__, __ge__, __eq__


class AbstractCustomConstraintsManager(ConstraintsManagerI):
    """
    An abstract custom constraints manager

    implements the functionality of a constraints manager for
    creating custom linear constraints and variables (not Gurobi constraints)
    for verifying an agent-environment system.

    Includes a variable tracker as a subclass of VariableTrackerManager
    """

    VAR_NAME_PREFIX = "C"

    def __init__(self):

        super(AbstractCustomConstraintsManager, self).__init__()

        self.var_counter = 0

        # not used anymore,
        # in the depth first way of computing problems, it accumulates too many variables
        self._vars = []

        self.stats = ExecutionStatistics(0, 0, 0)
        self.variable_tracker = AESVariableTracker()

    def _get_next_var_name(self):
        # :side-effects: Modifies self.var_counter
        self.var_counter += 1

        return self.VAR_NAME_PREFIX + "{}".format(self.var_counter-1)

    def _add_continuous_variable(self, lower, upper):
        var = ContinuousVariable(self._get_next_var_name(), lower, upper)
        # self._vars.append(var)
        return var

    def _add_binary_variable(self, lower, upper):
        var = BinaryVariable(self._get_next_var_name(), lower, upper)
        # self._vars.append(var)
        return var

    def _add_integer_variable(self, lower, upper):
        var = IntegerVariable(self._get_next_var_name(), lower, upper)
        # self._vars.append(var)
        return var

    # to be used by MILPEncoder
    def addVar(self, vtype=None, lb=None, ub=None):
        if vtype is None or vtype == GRB.CONTINUOUS:
            lower = 0 if lb is None else lb
            upper = float("inf") if ub is None else ub
            return self._add_continuous_variable(lower, upper)

        if vtype == GRB.BINARY:
            lower = 0 if lb is None else lb
            upper = 1 if ub is None else ub
            return self._add_binary_variable(lower, upper)

        if vtype == GRB.INTEGER:
            lower = 0 if lb is None else lb
            upper = sys.maxsize if ub is None else ub
            return self._add_integer_variable(lower, upper)

        raise Exception("Unsupported variable type", vtype)

    """
    Implementation of the abstract methods from the abstract constraints manager class
    """
    def create_state_variables(self, var_number, lbs=None, ubs=None):
        if lbs is None or ubs is None:
            vars = [ContinuousVariable(self._get_next_var_name(),
                                       float("-inf"), float("inf")) for _ in range(var_number)]
        else:
            vars = [ContinuousVariable(self._get_next_var_name(),
                                       lbs[i], ubs[i]) for i in range(var_number)]

        # self._vars.extend(vars)

        return vars

    def create_binary_variables(self, var_number, lbs=None, ubs=None):
        if lbs is None or ubs is None:
            vars = [BinaryVariable(self._get_next_var_name()) for _ in range(var_number)]
        else:
            vars = [BinaryVariable(self._get_next_var_name(), lb=lbs[i], ub=ubs[i]) for i in range(var_number)]

        # self._vars.extend(vars)

        return vars

    def create_integer_variable(self, lb, ub):
        var = IntegerVariable(self._get_next_var_name(), lb=lb, ub=ub)
        return var

    @staticmethod
    def get_sum_constraint(vars, rhs):
        coeffs = [1 for _ in range(len(vars))]
        lhs = WeightedSum(vars, coeffs)
        return LinearConstraint(lhs, __eq__, rhs)

    @staticmethod
    def get_assignment_constraint(var, value):
        lhs = WeightedSum([var], [1])
        return LinearConstraint(lhs, __eq__, value)

    @staticmethod
    def get_equality_constraint(var1, var2):
        lhs = WeightedSum([var1, var2], [1, -1])
        return LinearConstraint(lhs, __eq__, 0)

    @staticmethod
    def create_indicator_constraint(binvar, binval, expr):
        return IndicatorConstraint(binvar, binval, expr)

    @staticmethod
    def get_le_constraint(var, value):
        lhs = WeightedSum([var], [1])
        return LinearConstraint(lhs, __le__, value)

    @staticmethod
    def get_ge_constraint(var, value):
        lhs = WeightedSum([var], [1])
        return LinearConstraint(lhs, __ge__, value)

    @staticmethod
    def get_inequality_constraint(var, value, inequality):
        lhs = WeightedSum([var], [1])
        return LinearConstraint(lhs, inequality, value)

    @staticmethod
    def get_linear_constraint(vars, coeff, rhs, sense=__eq__):
        """
        :param vars: a list of variables
        :param coeff: a list of numbers
        :param rhs: a number
        :param sense:
        :return:
        """
        lhs = WeightedSum(vars, coeff)
        return LinearConstraint(lhs, sense, rhs)

    @staticmethod
    def get_max_constraint(var, values):
        return MaxConstraint(var, values)

    @staticmethod
    def get_min_constraint(var, values):
        return MinConstraint(var, values)

    def encode_max_constraint(self, y, x, c, M):
        """
        Encoding y = max(x,c) where x is a variable, c is a constant, M is the big-M constant
        :return: constraints encoding the restriction
        """
        [delta] = self.create_binary_variables(1)

        return [
            # y >= x
            self.get_linear_constraint([y,x], [1,-1], 0, sense=__ge__),
            # y >= c
            self.get_ge_constraint(y, c),
            # y <= x + M * (1 - delta) === y - x + M*delta <= M
            self.get_linear_constraint([y, x, delta], [1, -1, M], M, sense=__le__),
            # y <= c + M * delta === y - M*delta <= c
            self.get_linear_constraint([y, delta], [1, -M], c, sense=__le__)
        ]

    def encode_min_constraint(self, y, x, c, M):
        """
        Encoding y = min(x,c) where x is a variable, c is a constant
        :return: constraints encoding the restriction
        """
        [delta] = self.create_binary_variables(1)

        return [
            # y <= x
            self.get_linear_constraint([y,x], [1,-1], 0, sense=__le__),
            # y <= c
            self.get_le_constraint(y, c),
            # y >= x - M(1 - delta) === y - x - M*delta >= -M
            self.get_linear_constraint([y, x, delta], [1, -1, -M], -M, sense=__ge__),
            # y >= c - M * delta === y + M*delta >= c
            self.get_linear_constraint([y, delta], [1, M], c, sense=__ge__)
        ]

    @staticmethod
    def get_atomic_constraint(constraint, state_vars):
        if hasattr(constraint, "get_custom_atomic_constraint"):
            return constraint.get_custom_atomic_constraint(state_vars)
        else:
            raise Exception("Unexpected type of atomic constraint", constraint)

    def update(self):
        pass

    def get_variable_tracker(self):
        return self.variable_tracker




