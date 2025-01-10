from gurobipy import *

from src.utils.constraint import WeightedSum, LinearConstraint
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.constrmanager.abstract_custom_constraints_manager import AbstractCustomConstraintsManager
from src.verification.constrmanager.execution_statistics import ExecutionStatistics
from src.verification.constrmanager.gurobi_feasibility_checker import GurobiFeasibilityChecker
from src.verification.constrmanager.variable_bounds_manager import VariableBoundsManager
from src.verification.nnencoder.custommilpencoder import CustomMILPEncoder
from src.verification.bounds.boundsmanager import BoundsManager
from operator import __le__, __ge__, __eq__


class CustomConstraintsManager(AbstractCustomConstraintsManager):
    """
        An implementation of constraint manager that
        creates custom linear constraints and variables (not Gurobi constraints)
    """

    def __init__(self):
        super(CustomConstraintsManager, self).__init__()

        self.variable_bounds_manager = VariableBoundsManager()

        self.stats = ExecutionStatistics(0, 0, 0)

    def check_feasibility(self, gmodel, trace_variables=None):
        return GurobiFeasibilityChecker.check_feasibility(gmodel, self.stats, self.variable_tracker, trace_variables=trace_variables)

    def create_gmodel(self, constraints, trace):
        """
        Creates a Gurobi model from a set of custom constraints.

        :param constraints: the actual constraints constituting the linear program
        :param trace: variables that should be in the model (even if some of them
            do not appear in the constraints), expected to be a pair of arrays of arrays.
        :return:
        """
        gmodel = Model("AES")
        gmodel.Params.LogToConsole = 0

        # Create a dictionary of gurobi variables for
        # easy conversion of custom constraints to gurobi constraints
        all_grb_vars = self.add_variables_to_gmodel(constraints, trace[0] + trace[1], gmodel)

        for constr in constraints:
            gmodel.addConstr(constr.to_gurobi_constr(all_grb_vars, gmodel))

        return gmodel

    def add_variables_to_gmodel(self, constraints, variables, gmodel):
        # First compute the set of variables used in the given set of constraints
        used_vars = {item for c in constraints for item in c.get_vars()}

        # Second add the provided variables (some of them might not be used in the constraints)
        used_vars.update({var for vars in variables for var in vars})

        # Third, create gurobi variables and add them to a dictionary
        all_grb_vars = {}
        for var in used_vars:
            grb_var = gmodel.addVar(**var.to_gurobi_var())
            all_grb_vars[var.varName] = grb_var

        # Return the dictionary for future use
        return all_grb_vars

    def add_variable_bounds(self, vars, bounds):
        self.variable_bounds_manager.add_variable_bounds(vars, bounds)

    def get_variable_bounds(self, vars):
        return self.variable_bounds_manager.get_variable_bounds(vars)

    def get_stats(self):
        return self.stats

    def get_network_constraints(self, layers, input_vars, store_q_vars=False):
        # :side-effects: updates self.var_counter

        input_hyper_rect = self.get_variable_bounds(input_vars)

        # to follow more functional style we create objects only when needed
        bounds_manager = BoundsManager()
        numeric_preact_bounds, numeric_postact_bounds = bounds_manager.compute_bounds(layers, input_hyper_rect)

        ffnn_encoder = CustomMILPEncoder(numeric_preact_bounds)

        # the encoder takes the computed bounds into account
        dense, relu = ffnn_encoder.add_vars(self, layers)

        # Due to hack, set norm flag to assume first activation function is linear.
        q_vars, constrs = ffnn_encoder.get_constraints(layers, input_vars, dense, relu)

        if store_q_vars:
            self.variable_tracker.add_q_variables_to_tracker(q_vars)

        self.add_variable_bounds(q_vars, numeric_preact_bounds[-1])

        return q_vars, constrs

    def get_argmax_constraints(self, q_vars, use_q_bounds=True):
        # :side-effects: updates variables counter

        l = len(q_vars)

        constrs_to_add = []

        lower = [0 for _ in range(l)]
        upper = [1 for _ in range(l)]

        if use_q_bounds:
            q_bounds = self.get_variable_bounds(q_vars)

            # tightening the bounds for the binary variables by taking into
            # account the q bounds
            for i in range(l):
                other_upper = [q_bounds.get_upper()[j] for j in range(l) if j is not i]

                # if we know for sure that i-th output is the maximum
                if q_bounds.get_lower()[i] > max(other_upper):
                    for j in range(l):
                        if i == j:
                            continue
                        upper[j] = 0
                    lower[i] = 1
                    break

                # otherwise we can only determine what outputs are not maximums
                for j in range(i, l):
                    if q_bounds.get_lower()[i] > q_bounds.get_upper()[j]:
                        upper[j] = 0
                    elif q_bounds.get_lower()[j] > q_bounds.get_upper()[i]:
                        upper[i] = 0

        # Initialise N binary variables, where N is the number of q-values.
        o = self.create_binary_variables(l, lbs=lower, ubs=upper)

        self.add_variable_bounds(o, HyperRectangleBounds(lower,upper))

        # For each output dimension, add constraints relating to it being maximal
        for i in range(l):
            if upper[i] == 0:
                continue

            for j in range(l):
                if i == j:
                    continue

                # q_vars[i] >= q_vars[j]
                linconstr = LinearConstraint(
                    WeightedSum([q_vars[i], q_vars[j]], [1, -1]),
                    __ge__,
                    0
                )
                constrs_to_add.append(
                    self.create_indicator_constraint(o[i], 1, linconstr)
                )

        # Ensure that at most one o[i] is true
        constrs_to_add.append(
            self.get_sum_constraint(o, 1)
        )

        return o, constrs_to_add

    def get_argmax_index_constraints(self, binary_vars):
        # :side-effects: updates self.var_counter
        l = len(binary_vars)

        # attempt to get better lower and upper bounds for the integer output
        lower = 0
        upper = l - 1
        for i in range(l):
            bvar = binary_vars[i]
            if bvar.ub == 0:
                lower += 1
            else:
                break

        for i in range(l)[::-1]:
            bvar = binary_vars[i]
            if bvar.ub == 0:
                upper -= 1
            else:
                break

        constrs_to_add = []

        o = self._add_integer_variable(lower, upper)
        for i in range(l):
            constrs_to_add.append(
                self.create_indicator_constraint(binary_vars[i], 1,
                                                 self.get_assignment_constraint(o, i))
            )

        return o, constrs_to_add

