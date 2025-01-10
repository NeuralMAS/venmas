import numpy as np
from gurobipy import max_, min_

from src.utils.constraint_interface import Constraint
from operator import __le__, __ge__, __eq__

from src.utils.variable import BinaryVariable
from src.utils.variable_interface import Variable


class LinearConstraint(Constraint):

    def __init__(self, lhs, sense, rhs, name=""):
        """

        :param lhs: Left-hand side for the new constraint,
        a weighted sum of variables.
        :param sense: Sense for the new constraint.
        :param rhs: Right-hand side for the new constraint,
        a constant.
        :param name: Name for new constraint.
        """

        assert isinstance(lhs, WeightedSum)

        self.name = name
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs
        self.sense_repr = {__eq__: "==", __le__: "<=", __ge__: ">="}[sense]

    def accept(self, visitor):
        pass

    def acceptI(self, visitor):
        pass

    def to_gurobi_constr(self, all_grb_vars, gmodel):
        vars = self.lhs.vars
        coeffs = np.array(self.lhs.coeffs)
        grb_vars = []
        for var in vars:
            # WARNING! var can be an int or float! Not necessarily varName.
            # Should be possible to do dot product on numbers.
            if hasattr(var, "varName"):
                grb_vars.append(all_grb_vars[var.varName])
            else:
                # WARNING! Assuming it is an int or float here.
                grb_vars.append(var)

            # This is wrong!
            # grb_vars.append(all_grb_vars[var.varName])
        grb_constr = self.sense(coeffs.dot(grb_vars), self.rhs)
        return grb_constr

    def get_vars(self):
        return self.lhs.vars

    def __str__(self):
        return "({} {} {})".format(self.lhs, self.sense_repr, self.rhs)

    def __repr__(self):
        return self.__str__()


class WeightedSum:
    def __init__(self, variables, coefficients):
        """

        :param variables: a list of variables
        :param coefficients: a list of matching length of coefficients
        """
        assert len(variables) == len(coefficients)

        self.vars = variables
        self.coeffs = coefficients
        self.var_coeff_map = {}
        for i in range(len(variables)):
            # WRONG! "variables" are not always strings! Could be an int/float!
            self.var_coeff_map[str(variables[i])] = coefficients[i]

    def __str__(self):
        return " + ".join(["{}*{}".format(coeff, varname) for varname, coeff in self.var_coeff_map.items()])

    def __repr__(self):
        return self.__str__()


class IndicatorConstraint(Constraint):

    def __init__(self, binvar, binval, lin_constr, name=""):
        """
        Add a new indicator constraint.
        :param binvar: A binary indicator variable.
        :param binval: The value for the binary indicator variable that would force the linear constraint to be satisfied.
        :param linear_constr: Linear constraint triggered by the indicator.
        :param name: Name for the new indicator constraint.
        """

        assert isinstance(lin_constr, LinearConstraint)
        # assert isinstance(binvar, BinaryVariable)

        self.name = name
        self.lin_constr = lin_constr
        self.binval = binval
        self.binvar = binvar

    def accept(self, visitor):
        pass

    def acceptI(self, visitor):
        pass

    def to_gurobi_constr(self, all_grb_vars, gmodel):
        grb_bin_var = all_grb_vars[self.binvar.varName]
        grb_lin_constr = self.lin_constr.to_gurobi_constr(all_grb_vars, gmodel)
        grb_constr = (grb_bin_var == self.binval) >> grb_lin_constr
        return grb_constr

    def get_vars(self):
        return [self.binvar] + self.lin_constr.get_vars()

    def __str__(self):
        return "({} == {}) >> {}".format(self.binvar, self.binval, self.lin_constr)

    def __repr__(self):
        return self.__str__()


class MaxConstraint(Constraint):

    def __init__(self, var, values, name=""):
        """
        Add a new max constraint.
        :param var: A variable which will be assigned the maximum value from the array of values.
        :param values: The values from which to select the maximum.
        :param name: Name for the new max constraint.
        """

        assert isinstance(values, list)
        assert isinstance(var, Variable)

        self.name = name
        self.var = var
        self.values = values

    def accept(self, visitor):
        pass

    def acceptI(self, visitor):
        pass

    def to_gurobi_constr(self, all_grb_vars, gmodel):
        grb_var = all_grb_vars[self.var.varName]
        grb_values = [all_grb_vars[val.varName] if isinstance(val, Variable) else val for val in self.values]
        grb_constr = (grb_var == max_(grb_values))
        return grb_constr

    def get_vars(self):
        return [self.var] + [val for val in self.values if isinstance(val, Variable)]

    def __str__(self):
        return "{} == max_({})".format(self.var, self.values)

    def __repr__(self):
        return self.__str__()


class MinConstraint(Constraint):

    def __init__(self, var, values, name=""):
        """
        Add a new max constraint.
        :param var: A variable which will be assigned the minimum value from the array of values.
        :param values: The values from which to select the minimum.
        :param name: Name for the new max constraint.
        """

        assert isinstance(values, list)
        assert isinstance(var, Variable)

        self.name = name
        self.var = var
        self.values = values

    def accept(self, visitor):
        pass

    def acceptI(self, visitor):
        pass

    def to_gurobi_constr(self, all_grb_vars, gmodel):
        grb_var = all_grb_vars[self.var.varName]
        grb_values = [all_grb_vars[val.varName] if isinstance(val, Variable) else val for val in self.values]
        grb_constr = (grb_var == min_(grb_values))
        return grb_constr

    def get_vars(self):
        return [self.var] + [val for val in self.values if isinstance(val, Variable)]

    def __str__(self):
        return "{} == min_({})".format(self.var, self.values)

    def __repr__(self):
        return self.__str__()
