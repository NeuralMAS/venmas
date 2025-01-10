import numpy as np


class Layer(object):
    """
    A class representing a layer of a neural network.
    This class is not associated with any activation function.
    """
    def __init__(self, output_shape, weights, bias, depth):
        self.output_shape = output_shape
        self.weights = weights
        self.bias = bias
        self.depth = depth

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias


class Relu(Layer):
    """
    A class representing a layer of a neural network with ReLU activation function.
    """
    def __init__(self, output_shape, weights, bias, depth):
        super().__init__(output_shape, weights, bias, depth)

    def clone(self):
        return Relu(self.output_shape, self.weights, self.bias, self.depth)

    def forward(self, inputs):
        return np.maximum(self.weights.dot(inputs) + self.bias, 0)


class Linear(Layer):
    """
    A class representing a layer of a neural network with Linear activation function.
    """
    def __init__(self, output_shape, weights, bias, depth):
        super().__init__(output_shape, weights, bias, depth)

    def clone(self):
        return Linear(self.output_shape, self.weights, self.bias, self.depth)

    def forward(self, inputs):
        return self.weights.dot(inputs) + self.bias
