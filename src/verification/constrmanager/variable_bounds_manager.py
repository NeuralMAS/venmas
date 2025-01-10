from src.verification.bounds.bounds import HyperRectangleBounds


class VariableBoundsManager:
    def __init__(self):
        self.variable_bounds_map = {}

    def add_variable_bounds(self, vars, bounds):
        """
        Method to store the bounds of variables in a dictionary to be used later.
        :param vars: a collection of variables
        :param bounds: a HyperRectangleBounds object storing bounds for the variables
        :return:
        """

        for i in range(len(vars)):
            var = vars[i]
            self.variable_bounds_map[str(var)] = bounds.get_dimension_bounds(i)

    def get_variable_bounds(self, vars):
        """
        Method to retrieve the bounds of variables from the dictionary.
        :param vars: a collection of Gurobi variables
        :param already_set: a flag indicating whether the variable bounds can be found in the variable itself
        :return: a HyperRectangleBounds object with bounds for the variables
        """

        lower = []
        upper = []
        for var in vars:
            l, u = self.variable_bounds_map[str(var)]
            lower.append(l)
            upper.append(u)

        return HyperRectangleBounds(lower, upper)
