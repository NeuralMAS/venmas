from src.network_parser.keras_parser import KerasParser
from src.network_parser.rnn_abstractor import RNNAbstractor
from src.utils.constants import RNNUnrollingMethods


class RecurrentNetworkModel:
    """
    A class for an internal representation of a recurrent neural network.
    Consists of three layers, each storing the relevant parameters
    such as weights, bias
    """
    def __init__(self):

        super(RecurrentNetworkModel, self).__init__()

        self.input_hidden = None
        self.hidden_hidden = None
        self.hidden_output = None
        self.n_layers = 0
        self.input_size = 0

    def parse(self, nn_filename):
        if nn_filename.endswith(".h5"):
            layers, n_layers, input_shape = KerasParser.parse_file(nn_filename)

            if n_layers != 3:
                raise Exception("Currently supporting only single layer recurrent neural networks"
                                "with one input, one hidden and one output layers only. "
                                "Got a network with {} layers".format(n_layers))

            self.input_hidden = layers[0]
            self.hidden_hidden = layers[1]
            self.hidden_output = layers[2]
            self.n_layers = n_layers
            self.input_size = input_shape

        else:
            raise Exception("Unsupported network model file format", nn_filename)

    def convert(self, keras_model):
        layers, n_layers, input_shape = KerasParser.parse_model(keras_model)

        if n_layers != 3:
            raise Exception("Currently supporting only single layer recurrent neural networks"
                            "with one input, one hidden and one output layers only. "
                            "Got a network with {} layers".format(n_layers))

        self.input_hidden = layers[0]
        self.hidden_hidden = layers[1]
        self.hidden_output = layers[2]
        self.n_layers = n_layers
        self.input_size = input_shape

    def unroll(self, depth, unrolling_method=None):
        if unrolling_method is None:
            if self.get_input_size() > self.get_hidden_size():
                unrolling_method = RNNUnrollingMethods.INPUT_ON_START
            else:
                unrolling_method = RNNUnrollingMethods.INPUT_ON_DEMAND

        rnn_abstractor = RNNAbstractor()
        return rnn_abstractor.build_abstraction(self, depth, unrolling_method)

    def get_input_hidden(self):
        return self.input_hidden

    def get_hidden_hidden(self):
        return self.hidden_hidden

    def get_hidden_output(self):
        return self.hidden_output

    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_hidden.output_shape