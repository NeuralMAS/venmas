import tensorflow.keras as keras
import numpy as np

from src.network_parser.layers import Relu, Linear
from src.utils.constants import ActivationFunctions


class KerasParser:

    @staticmethod
    def parse_file(nn_filename):
        nmodel = keras.models.load_model(nn_filename)
        return KerasParser.parse_model(nmodel)

    @staticmethod
    def parse_model(nmodel):
        layers = []
        keras_layer_counter = 0
        layer_counter = 1

        # Initialise the layers on the network in our format
        while keras_layer_counter < len(nmodel.layers):

            keras_layer = nmodel.layers[keras_layer_counter]

            # currently supporting only Dense and SimpleRNN layers
            if isinstance(keras_layer, keras.layers.Dense):
                keras_layer_counter, layer = \
                    KerasParser._parse_keras_dense_layer(nmodel, keras_layer_counter, layer_counter)

                layers.append(layer)
                keras_layer_counter += 1
                layer_counter += 1
            elif isinstance(keras_layer, keras.layers.SimpleRNN) \
                    or isinstance(keras_layer, RNNCell):
                keras_layer_counter, input_hidden, hidden_hidden = \
                    KerasParser._parse_keras_simplernn_layer(nmodel, keras_layer_counter, layer_counter)

                layers.append(input_hidden)
                layers.append(hidden_hidden)
                keras_layer_counter += 1
                layer_counter += 2
            elif isinstance(keras_layer, keras.layers.InputLayer):
                keras_layer_counter += 1
            else:
                raise Exception("Unsupported network layer: {}.\n "
                                "Expected a Dense or a SimpleRNN layer.".format(keras_layer))

        n_layers = layer_counter - 1
        input_shape = layers[0].get_weights().shape[1]

        return layers, n_layers, input_shape

    @staticmethod
    def _parse_keras_dense_layer(nmodel, keras_layer_counter, layer_counter):
        """
        Parse a Dense layer.
        Since our layers always include an activation function,
        while keras layers might have the actual activation in a separate layer,
        we might also have a look at the next layer. In the latter case we return an updated index i

        :param nmodel: keras model
        :param keras_layer_counter: the index of the current (dense) layer in nmodel
        :param layer_counter: the depth of the current layer in our internal representation
        :return:
            i+1 if the next layer is an activation layer, i otherwise,
            the layer corresponding to the dense layer
        """
        layer = nmodel.layers[keras_layer_counter]

        weights = layer.get_weights()[0].T
        bias = layer.get_weights()[1]
        output_shape = len(bias)

        # Variable storing the layer's activation function
        # Currently supporting only ReLU and Linear.
        activation = ActivationFunctions.UNKNOWN

        # detect the activation function
        if layer.activation == keras.activations.relu:
            activation = ActivationFunctions.RELU
        elif layer.activation == keras.activations.softmax:
            activation = ActivationFunctions.LINEAR
        elif layer.activation == keras.activations.linear:
            # Below we check for the relevant activation function
            # that could be encoded in the next keras layer
            keras_layer_counter, activation = KerasParser.check_activation_in_next_layer(nmodel, keras_layer_counter)

        # return depending on the value of activation
        if activation == ActivationFunctions.RELU:
            return keras_layer_counter, Relu(output_shape, weights, bias, layer_counter)
        elif activation == ActivationFunctions.LINEAR:
            return keras_layer_counter, Linear(output_shape, weights, bias, layer_counter)
        else:
            raise Exception("Unsupported activation function", layer.activation)

    @staticmethod
    def _parse_keras_simplernn_layer(nmodel, keras_layer_counter, layer_counter):
        """
        Parse a SimpleRNN layer.
        Since our layers always include an activation function,
        while keras layers might have the actual activation in a separate layer,
        we might also have a look at the next layer. In the latter case we return an updated index i

        :param nmodel: keras model
        :param keras_layer_counter: the index of the current (dense) layer in nmodel
        :param layer_counter: the depth of the current layer in our internal representation
        :return:
            i+1 if the next layer is an activation layer, i otherwise,
            layer corresponding to the input-to-hidden connection,
            layer corresponding to the hidden-to-hidden connection
        """
        layer = nmodel.layers[keras_layer_counter]

        output_shape = layer.output_shape[1]

        # the first component of the weights are the weights from the input to the hidden layer
        # the second component of the weights are the weights from the hidden layer to itself
        # there are no biases
        input_hidden = layer.get_weights()[0].T
        hidden_hidden = layer.get_weights()[1].T

        # Variable storing the layer's activation function
        # Currently supporting only ReLU and Linear.
        activation = ActivationFunctions.UNKNOWN

        # detect the activation function
        if layer.activation == keras.activations.relu:
            activation = ActivationFunctions.RELU

        elif layer.activation == keras.activations.linear:
            # Below we check for the relevant activation function
            # that could be encoded in the next keras layer
            keras_layer_counter, activation = KerasParser.check_activation_in_next_layer(nmodel, keras_layer_counter)

        # return depending on the value of activation
        if activation == ActivationFunctions.RELU:
            return keras_layer_counter, \
                   Relu(output_shape, input_hidden, np.zeros(input_hidden.shape[0]), layer_counter), \
                   Relu(output_shape, hidden_hidden, np.zeros(hidden_hidden.shape[0]), layer_counter + 1)
        elif activation == ActivationFunctions.LINEAR:
            return keras_layer_counter, \
                   Linear(output_shape, input_hidden, np.zeros(input_hidden.shape[0]), layer_counter), \
                   Linear(output_shape, hidden_hidden, np.zeros(hidden_hidden.shape[0]), layer_counter + 1)
        else:
            raise Exception("Unsupported activation function", layer.activation, "in layer", keras_layer_counter)

    @staticmethod
    def check_activation_in_next_layer(nmodel, keras_layer_counter):
        """
        Method for checking if the activation function is stored in the next keras layer.
        Should only be called when checking a layer with linear activation.
        If it finds that the next layer is an Activation layer,
        updates the index of the current keras layer
        :param keras_layer_counter: the index of the current (non Activation) layer
        :param nmodel: keras model
        :return:
        """

        assert nmodel.layers[keras_layer_counter].activation == keras.activations.linear
        activation = ActivationFunctions.LINEAR

        if keras_layer_counter + 1 < len(nmodel.layers):
            if isinstance(nmodel.layers[keras_layer_counter + 1], keras.layers.Activation):
                keras_layer_counter += 1
                layer = nmodel.layers[keras_layer_counter]

                if layer.activation == keras.activations.relu:
                    activation = ActivationFunctions.RELU
                elif layer.activation == keras.activations.softmax:
                    # We can also accept softmax as we can compute argmax
                    # using MILP constraints
                    activation = ActivationFunctions.LINEAR
                elif layer.activation != keras.activations.linear:
                    raise Exception("Unsupported activation function", layer.activation, "in layer", keras_layer_counter)

        return keras_layer_counter, activation


# A hack
# A custom recurrent layer used in the network trained by us
class RNNCell(keras.layers.Layer):
    """Wrapped to fix call signature for super cell class."""
    def __init__(self, units, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build the layer."""
        self.kernel = self.add_weight(shape=(input_shape[0][-1], self.units),
                                      name='kernel',
                                      initializer='glorot_uniform')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer='identity',
            regularizer=None,
            constraint=None)
        self.bias = self.add_weight(shape=(self.units,),
                                    name='bias',
                                    initializer='zeros')
        self.activation = keras.activations.relu
        super().build(input_shape)

    def call(self, inputs):
        """Delegate assuming inputs[1] is states."""
        assert len(inputs) == 2
        state, hm1 = inputs
        h = keras.backend.dot(state, self.kernel) + keras.backend.dot(hm1, self.recurrent_kernel)
        # h = K.dot(state, self.kernel)
        h = keras.backend.bias_add(h, self.bias)
        return keras.backend.relu(h)

    def compute_output_shape(self, input_shape):
        """Compute the output shape list."""
        return input_shape[1]


