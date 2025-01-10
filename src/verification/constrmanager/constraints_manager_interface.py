from abc import ABCMeta, abstractmethod

NOT_IMPLEMENTED = "Needs to be implemented."


class ConstraintsManagerI:
    """
    Interface defining methods for a general constraints manager.
    """

    __metaclass__ = ABCMeta

    """
    Methods for creating variables
    """
    @abstractmethod
    def create_state_variables(self, var_number, lbs=None, ubs=None):
        return NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def create_binary_variables(self, var_number):
        return NotImplementedError(NOT_IMPLEMENTED)

    """
    Shortcuts for creating popular kinds of constraints
    """
    @staticmethod
    def get_sum_constraint(delta, rhs):
        return NotImplementedError(NOT_IMPLEMENTED)

    @staticmethod
    def get_assignment_constraint(var, value):
        return NotImplementedError(NOT_IMPLEMENTED)

    @staticmethod
    def get_equality_constraint(var1, var2):
        return NotImplementedError(NOT_IMPLEMENTED)

    @staticmethod
    def create_indicator_constraint(binvar, binval, expr):
        return NotImplementedError(NOT_IMPLEMENTED)

    @staticmethod
    def get_le_constraint(var, value):
        return NotImplementedError(NOT_IMPLEMENTED)

    @staticmethod
    def get_ge_constraint(var, value):
        return NotImplementedError(NOT_IMPLEMENTED)

    @staticmethod
    def get_inequality_constraint(var, value, inequality):
        return NotImplementedError(NOT_IMPLEMENTED)

    @staticmethod
    def get_linear_constraint(vars, coeff, rhs):
        return NotImplementedError(NOT_IMPLEMENTED)

    @staticmethod
    def get_max_constraint(var, values):
        return NotImplementedError(NOT_IMPLEMENTED)

    @staticmethod
    def get_min_constraint(var, values):
        return NotImplementedError(NOT_IMPLEMENTED)
