from gurobipy import *
from keras.src.layers import Dense

from src.network_parser.layers import Relu, Linear


class MILPEncoder:
    """
    The encoder that takes as input precomputed bounds of the network,
    works both for Gurobi model and for our custom model.
    """
    def __init__(self, precomputed_bounds):
        """
        :param precomputed_bounds: is a list of preactivation bounds layer by layer
        """
        self.precomputed_bounds = precomputed_bounds

    @staticmethod
    def _dense_vars(var_manager, layer, bounds):
        # :side-effects: Changes var_manager (can be a Gurobi model or our custom constraints manager).

        lower_bounds = bounds.get_lower()
        upper_bounds = bounds.get_upper()

        output_vars = []
        delta_vars = []
        for i in range(layer.output_shape):
            output_vars.append(var_manager.addVar(lb=lower_bounds[i], ub=upper_bounds[i]))
            delta_vars.append(var_manager.addVar(vtype=GRB.BINARY))
        return (output_vars, delta_vars)

    @staticmethod
    def _relu_vars(var_manager, layer, bounds):
        # :side-effects: Changes var_manager (can be a gmodel or a custom constraints manager).

        lower_bounds = bounds.get_lower()
        upper_bounds = bounds.get_upper()

        relu_vars = []
        for i in range(layer.output_shape):
            relu_vars.append(var_manager.addVar(lb=max(lower_bounds[i], 0), ub=max(upper_bounds[i], 0)))
        return relu_vars

    def add_vars(self, var_manager, layers):
        # :side-effects: Changes var_manager (can be a gmodel or a custom constraints manager).

        dense, relu = [], []
        for i in range(0, len(layers)):
            dense.append(self._dense_vars(var_manager, layers[i], self.precomputed_bounds[i]))

            if isinstance(layers[i], Relu):
                relu.append(self._relu_vars(var_manager, layers[i], self.precomputed_bounds[i]))
            elif isinstance(layers[i], Linear):
                relu.append([])
            else:
                raise Exception("Unsupported layer: {}. Expecting a Relu or a Linear layer".format(layers[i]))

        return (dense, relu)

    @staticmethod
    def _dense_constraints(layer, inputs, outputs):
        weights = layer.get_weights()
        bias = layer.get_bias()
        dotted_outputs = []

        constrs = []
        for i in range(layer.output_shape):
            dotted_outputs.append(weights[i].dot(inputs) + bias[i])
            constrs.append(dotted_outputs[i] - outputs[i] == 0)
        return constrs

    """
    Functionality for getting Gurobi constraints
    """
    @staticmethod
    def _relu_constraints(layer, pre, post, delta, bounds):
        lower_bounds = bounds.get_lower()
        upper_bounds = bounds.get_upper()

        constrs = []

        for i in range(layer.output_shape):
            if lower_bounds[i] >= 0:
                constrs.append(post[i] == pre[i])
                constrs.append(delta[i] == 1)
            elif upper_bounds[i] <= 0:
                constrs.append(post[i] == 0)
                constrs.append(delta[i] == 0)
            else:
                constrs.append(post[i] >= pre[i])
                constrs.append(post[i] <= pre[i] - lower_bounds[i] * (1 - delta[i]))
                constrs.append(post[i] <= upper_bounds[i] * delta[i])

        return constrs

    def get_constraints(self, layers, input, dense, activation):
        """
        A method that returns a set of MILP constraints encoding
        a neural network, given its layers and the variables
        :param layers: the layers of the network
        :param input: the input variables
        :param dense: list of pairs (o,d) where o are the variables for the weighted sum and
        d are the variables for encoding the activation function, for Relu, they are assumed to be binary
        :param activation: the variables for the output of activations
        :return: the output variables and the resulting MILP constraints
        """
        constrs = []
        o = None

        il = input
        for i in range(0, len(layers)):
            o, d = dense[i]
            r = activation[i]

            constrs.extend(self._dense_constraints(layers[i], il, o))

            if isinstance(layers[i], Relu):
                constrs.extend(self._relu_constraints(layers[i], o, r, d, self.precomputed_bounds[i]))
            elif not isinstance(layers[i], Linear):
                raise Exception("Unsupported layer: {}. Expecting a Relu or a Linear layer".format(layers[i]))

            il = r

        return o, constrs

