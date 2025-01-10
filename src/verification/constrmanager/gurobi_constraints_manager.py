from gurobipy import *

from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.bounds.boundsmanager import BoundsManager
from src.verification.constrmanager.execution_statistics import ExecutionStatistics
from src.verification.constrmanager.gurobi_feasibility_checker import GurobiFeasibilityChecker
from src.verification.constrmanager.variable_bounds_manager import VariableBoundsManager
from src.verification.nnencoder.milpencoder import MILPEncoder
from src.verification.constrmanager.abstract_gurobi_constraints_manager import AbstractGurobiConstraintsManager


class GurobiConstraintsManager(AbstractGurobiConstraintsManager):
    """
        The default implementation of Gurobi constraint manager
    """

    def __init__(self, gmodel):
        super(GurobiConstraintsManager, self).__init__(gmodel)

        self.stats = ExecutionStatistics(0, 0, 0)
        self.variable_bounds_manager = VariableBoundsManager()

    def __str__(self):
        return "Gurobi Constraints Manager"

    def copy(self):
        # :side-effects: Creates a new Gurobi model and creates a new ConstraintsManager.

        # Due to the lazy update approach in Gurobi, you have to call update before copying it.
        self.gmodel.update()
        gmodel_copy = self.gmodel.copy()

        return GurobiConstraintsManager(gmodel_copy)

    def get_stats(self):
        return self.stats

    def check_feasibility(self):
        # :side-effects: Updates, optimizes and reads from Gurobi model, modifies stats object.
        return GurobiFeasibilityChecker.check_feasibility(self.gmodel, self.stats, self.variable_tracker)

    def add_variable_bounds(self, vars, bounds):
        # :side-effects: Updates Gurobi model.
        self.gmodel.update()

        self.variable_bounds_manager.add_variable_bounds(vars, bounds)

    def get_variable_bounds(self, vars):
        # :side-effects: Updates Gurobi model.
        self.gmodel.update()

        return self.variable_bounds_manager.get_variable_bounds(vars)

    def get_network_constraints(self, layers, input_vars, store_q_vars=False):
        """
        An overridden implementation of the method from the super class
        to compute and to use bounds for the MILP encoding.
        """
        # :side-effects: Writes to Gurobi model.

        input_hyper_rect = self.get_variable_bounds(input_vars)

        # to follow more functional style we create objects only when needed
        bounds_manager = BoundsManager()
        numeric_preact_bounds, numeric_postact_bounds = bounds_manager.compute_bounds(layers, input_hyper_rect)

        ffnn_encoder = MILPEncoder(numeric_preact_bounds)
        dense, relu = ffnn_encoder.add_vars(self.gmodel, layers)
        q_vars, constrs = ffnn_encoder.get_constraints(layers, input_vars, dense, relu)

        if store_q_vars:
            """
            Store q-variables, useful for debugging.
            When the output of a network is argmax, these variables are not action, nor state variables,
            so they would not be tracked otherwise. 
            """
            self.variable_tracker.add_q_variables_to_tracker(q_vars)

        # MA: EB Please utilise comments.
        self.add_variable_bounds(q_vars, numeric_preact_bounds[-1])

        return q_vars, constrs

    def get_argmax_constraints(self, q_vars, use_q_bounds=True):
        """
        An overridden implementation of the method from the super class
        that potentially returns an optimised set of argmax constraints
        by taking into account the bounds of q_vars.
        """
        # :side-effects: Writes to Gurobi model.

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
        o = [self.gmodel.addVar(vtype=GRB.BINARY, lb=lower[i], ub=upper[i]) for i in range(l)]

        self.add_variable_bounds(o, HyperRectangleBounds(lower,upper))
        self.gmodel.update()

        # For each output dimension, add constraints relating to it being maximal
        for i in range(l):
            if upper[i] == 0:
                continue

            for j in range(l):
                if i == j:
                    continue
                constrs_to_add.append(self.create_indicator_constraint(o[i], 1, q_vars[i] >= q_vars[j]))
                # constrs_to_add.append((o[i] == 1) >> (q_vars[i] >= q_vars[j]))

        # Ensure that at most one o[i] is true
        constrs_to_add.append(quicksum(o) == 1)

        return o, constrs_to_add

    def get_argmax_index_constraints(self, binary_vars):
        """
        A method for generating constraints choosing the index of the binary variable that is true
        :param binary_vars: the binary variables out of which only one is true
        :return: an integer variable holding the index of the true binary variable,
            resulting MILP constraints and a collection of binary variables used in premises of
            indicator constraints
        """
        # :side-effects: Writes to Gurobi model.
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

        o = self.gmodel.addVar(vtype=GRB.INTEGER, lb=lower, ub=upper)
        self.gmodel.update()
        for i in range(l):
            constrs_to_add.append(
                self.create_indicator_constraint(binary_vars[i], 1, o == i)
            )

        return o, constrs_to_add

