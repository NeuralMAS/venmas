import numpy as np

from src.network_parser.layers import Relu, Linear


class NnetParser:

    @staticmethod
    def parse_file(nn_filename):
        file = open(nn_filename, 'r')

        """
        Skip comment lines
        """
        line = file.readline()
        while line[0] == '/':
            line = file.readline()

        """
        Read number of layers, input size and layers' sizes
        """
        data = line.strip()[0:-1].split(',')
        n_layers = int(data[0])
        input_shape = int(data[1])

        line = file.readline()
        data = line.strip()[0:-1].split(',')
        layer_sizes = [int(d) for d in data]

        """
        Skip minimum, maximum, mean and range values
        """
        for i in range(5):
            line = file.readline()

        layers = []
        for i in range(n_layers):
            layer = NnetParser._parse_layer(file, i+1, layer_sizes[i], layer_sizes[i+1], n_layers)
            layers.append(layer)

        return layers, n_layers, input_shape

    @staticmethod
    def _parse_layer(file, layer_counter, input_shape, output_shape, num_layers):

        weights = np.empty(shape=(output_shape, input_shape), dtype=np.float32)
        bias = np.empty(shape=output_shape, dtype=np.float32)

        """
        Read the weights
        """
        for row in range(output_shape):
            line = file.readline()
            data = line.strip()[0:-1].split(',')

            for column in range(input_shape):
                weights[row][column] = float(data[column])

        """
        Read the bias
        """
        for row in range(output_shape):
            line = file.readline()
            data = line.strip()[0:-1].split(',')

            bias[row] = float(data[0])

        """
        Construct the layer
        """
        # All intermediate layers are assumed to have ReLU activation
        # and the last one Linear
        if layer_counter < num_layers:
            return Relu(output_shape, weights, bias, layer_counter)
        elif layer_counter == num_layers:
            return Linear(output_shape, weights, bias, layer_counter)
        else:
            raise Exception("Wrong input parameters, the layer counter ({}) is greater "
                            "than the total number of layers ({})".format(layer_counter, num_layers))


