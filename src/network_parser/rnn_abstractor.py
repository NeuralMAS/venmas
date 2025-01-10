import numpy as np


from src.network_parser.layers import Relu, Linear
from src.network_parser.network_model import NetworkModel
from src.network_parser.recurrent_network_model import *
from src.utils.constants import RNNUnrollingMethods


class RNNAbstractor(object):
    """
    A class for computing unrollings of recurrent neural network
    to a specified depth.
    Two unrolling methods are possible:
        - input on start, which applies in the first layer
          the input-to-hidden weighted sum to each tuple in the input sequence, and

        - input on demand, which applies the input-to-hidden weighted sum to a tuple
          in the input sequence only when it is immediately needed.
    """

    def __init__(self):
        """
        """

    def build_abstraction(self, rnn, depth, unrolling_method):
        """
        :param rnn: an instance of RecurrentNeuralNetwork
        :param depth: the unrolling depth
        :param unrolling_method:
        :return: an instance of NetworkModel that corresponds to the unrolling of rnn
        to the specified depth
        """
        # if not isinstance(rnn, RecurrentNetworkModel):
        #     raise Exception("Only RecurrentNetworkModel instances are accepted. Got instead", rnn)
        if not depth >= 1:
            raise Exception("The unrolling depth must be a positive integer")

        if unrolling_method == RNNUnrollingMethods.INPUT_ON_START:
            return self._build_ios(rnn, depth)
        elif unrolling_method == RNNUnrollingMethods.INPUT_ON_DEMAND:
            return self._build_iod(rnn, depth)
        else:
            raise Exception("Unknown unrolling method {}. Expecting one of RNNUnrollingMethods.INPUT_ON_START, "
                            "RNNUnrollingMethods.INPUT_ON_DEMAND".format(unrolling_method))

    def _build_ios(self, rnn, depth):
        layers = []

        layers.append(self._build_ios_first_layer(rnn, depth))

        for l in range(1, depth):
            layers.append(self._build_ios_hidden_layer_at_step(rnn, depth, l))

        layers.append(self._build_last_layer(rnn, depth))

        input_shape = layers[0].get_weights().shape[1]
        return NetworkModel(layers, len(layers), input_shape)

    def _build_iod(self, rnn, depth):
        layers = []

        layers.append(self._build_iod_first_layer(rnn, depth))

        for i in range(1, depth):
            layers.append(self._build_iod_hidden_layer_at_step(rnn, depth, i))

        layers.append(self._build_last_layer(rnn, depth))

        input_shape = layers[0].get_weights().shape[1]
        return NetworkModel(layers, len(layers), input_shape)

    @staticmethod
    def _build_iod_first_layer(rnn, depth):
        ih_layer = rnn.get_input_hidden()

        """
        The weights of the first layer is a block diagonal matrix having the
        weights of the input_hidden layer in the first block, and
        the identity matrices in the remaining blocks (assuming depth=4):

            Wih  0   0   0
             0   I   0   0
             0   0   I   0
             0   0   0   I

        The width is depth * input_size
        The height is ih_height + (depth - 1) * input_size
        """

        input_size = rnn.get_input_size()
        hidden_size = rnn.get_hidden_size()
        identity_size = input_size * (depth - 1)

        # The first block column
        first_block_zeroes = np.zeros((identity_size, input_size))
        first_block = np.block([[ih_layer.get_weights()], [first_block_zeroes]])

        # The second block column (with the identity)
        second_block_zeroes = np.zeros((hidden_size, identity_size))
        second_block_identity = np.eye(identity_size)
        second_block = np.block([[second_block_zeroes], [second_block_identity]])

        weights = np.block([first_block, second_block])
        output_shape = weights.shape[0]
        bias = np.zeros(output_shape)

        if isinstance(ih_layer, Relu):
            return Relu(output_shape, weights, bias, 1)
        elif isinstance(ih_layer, Linear):
            return Linear(output_shape, weights, bias, 1)
        else:
            raise Exception("Unsupported layer", ih_layer)

    @staticmethod
    def _build_iod_hidden_layer_at_step(rnn, depth, step):
        """
        Assuming depth=4, the weights of the first hidden layer look as follows:

            Whh Wih  0   0
             0   0   I   0
             0   0   0   I

        Of the second hidden layer as follows:

            Whh Wih  0
             0   0   I

        And of the last hidden layer as:

            Whh Wih

        The width is hidden_size + (depth - l) * input_size
        The height is hidden_size + (depth - l - 1)* input_size
        """
        ih_layer = rnn.get_input_hidden()
        hh_layer = rnn.get_hidden_hidden()

        input_size = rnn.get_input_size()
        hidden_size = rnn.get_hidden_size()
        identity_size = input_size * (depth - step - 1)

        # The first block column
        first_block_zeroes = np.zeros((identity_size, hidden_size + input_size))
        first_block = np.block([[hh_layer.get_weights(), ih_layer.get_weights()],
                                [first_block_zeroes]])

        # The second block column (with the identity)
        second_block_zeroes = np.zeros((hidden_size, identity_size))
        second_block_identity = np.eye(identity_size)
        second_block = np.block([[second_block_zeroes], [second_block_identity]])

        weights = np.block([first_block, second_block])
        output_shape = weights.shape[0]
        bias = np.zeros(output_shape)

        if isinstance(ih_layer, Relu):
            return Relu(output_shape, weights, bias, step + 1)
        elif isinstance(ih_layer, Linear):
            return Linear(output_shape, weights, bias, step + 1)
        else:
            raise Exception("Unsupported layer", ih_layer)

    @staticmethod
    def _build_ios_first_layer(rnn, depth):
        ih_layer = rnn.get_input_hidden()

        """
        The weights of the first layer is a block matrix having the
        weights of the input_hidden layer on the diagonal (assuming depth=4):
        
            Wih  0   0   0
             0  Wih  0   0
             0   0  Wih  0
             0   0   0  Wih
             
        The width is depth * ih_width
        The height is depth * ih_height
        """
        weights = np.kron(np.eye(depth), ih_layer.get_weights())

        output_shape = weights.shape[0]
        bias = np.zeros(output_shape)
        if isinstance(ih_layer, Relu):
            return Relu(output_shape, weights, bias, 1)
        elif isinstance(ih_layer, Linear):
            return Linear(output_shape, weights, bias, 1)
        else:
            raise Exception("Unsupported layer", ih_layer)

    @staticmethod
    def _build_ios_hidden_layer_at_step(rnn, depth, l):
        hh_layer = rnn.get_hidden_hidden()

        hidden_size = hh_layer.output_shape
        hh_weights = hh_layer.get_weights()

        """
        Assuming depth=4, the weights of the first hidden layer look as follows:

            Whh  I   0   0
             0   0   I   0
             0   0   0   I

        Of the second hidden layer as follows:

            Whh  I   0 
             0   0   I 

        And of the last hidden layer as:

            Whh  I 

        The width is (depth - l + 1) * hidden_size
        The height is (depth - l)* hidden_size
        """

        # The first block column
        first_block_zeroes = np.zeros(((depth - l - 1) * hidden_size, hidden_size))
        first_block = np.block([[hh_weights], [first_block_zeroes]])

        # The second block column, an identity matrix
        second_block = np.eye((depth - l) * hidden_size)

        weights = np.block([first_block, second_block])
        output_shape = weights.shape[0]
        bias = np.zeros(output_shape)

        if isinstance(hh_layer, Relu):
            return Relu(output_shape, weights, bias, l + 1)
        elif isinstance(hh_layer, Linear):
            return Linear(output_shape, weights, bias, l + 1)
        else:
            raise Exception("Unsupported layer", hh_layer)

    @staticmethod
    def _build_last_layer(rnn, depth):
        ho_layer = rnn.get_hidden_output()

        weights = ho_layer.get_weights()
        bias = ho_layer.get_bias()
        output_shape = weights.shape[0]

        if isinstance(ho_layer, Relu):
            return Relu(output_shape, weights, bias, depth + 1)
        elif isinstance(ho_layer, Linear):
            return Linear(output_shape, weights, bias, depth + 1)
        else:
            raise Exception("Unsupported layer", ho_layer)


