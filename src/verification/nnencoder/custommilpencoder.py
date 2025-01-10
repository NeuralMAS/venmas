from src.utils.constraint import LinearConstraint, WeightedSum
from operator import __le__, __ge__, __eq__

from src.verification.nnencoder.milpencoder import MILPEncoder


class CustomMILPEncoder(MILPEncoder):
    """
    The encoder that generates custom constraints.
    Only overrides two methods of the super-class
    """
    def __init__(self, precomputed_bounds):
        """
        :param precomputed_bounds: is a list of preactivation bounds layer by layer
        """
        super(CustomMILPEncoder, self).__init__(precomputed_bounds)

    """
    Overriding the superclass methods to return custom constraints (not Gurobi ones)
    """
    def _dense_constraints(self, layer, inputs, outputs):
        weights = layer.get_weights()
        bias = layer.get_bias()

        constrs = []
        for i in range(layer.output_shape):

            vars = inputs + [outputs[i]]
            coeffs = list(weights[i]) + [-1]
            constrs.append(
                LinearConstraint(WeightedSum(vars, coeffs), __eq__, -bias[i])
            )
        return constrs

    @staticmethod
    def _relu_constraints(layer, pre, post, delta, bounds):
        lower_bounds = bounds.get_lower()
        upper_bounds = bounds.get_upper()

        constrs = []

        for i in range(layer.output_shape):
            if lower_bounds[i] >= 0:
                constrs.append(LinearConstraint(WeightedSum([post[i], pre[i]], [1, -1]), __eq__, 0))
                constrs.append(LinearConstraint(WeightedSum([delta[i]], [1]), __eq__, 1))
            elif upper_bounds[i] <= 0:
                constrs.append(LinearConstraint(WeightedSum([post[i]], [1]), __eq__, 0))
                constrs.append(LinearConstraint(WeightedSum([delta[i]], [1]), __eq__, 0))
            else:
                constrs.append(LinearConstraint(WeightedSum([post[i], pre[i]], [1, -1]), __ge__, 0))
                constrs.append(LinearConstraint(WeightedSum([post[i], pre[i], delta[i]], [1, -1, -lower_bounds[i]]), __le__, -lower_bounds[i]))
                constrs.append(LinearConstraint(WeightedSum([post[i], delta[i]], [1, -upper_bounds[i]]), __le__, 0))

        return constrs


