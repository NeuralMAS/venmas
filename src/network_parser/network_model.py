from src.network_parser.keras_parser import KerasParser
# from src.network_parser.nnet_parser import NnetParser


class NetworkModel(object):
    """
    A class for an internal representation of a feed-forward neural network.
    Consists of an array of layers, each storing the relevant parameters
    such as weights, bias
    """
    def __init__(self, layers=[], n_layers=0, input_shape=0):
        self.layers = layers
        self.n_layers = n_layers
        self.input_shape = input_shape

    def clone(self):
        new_model = NetworkModel()
        for layer in self.layers:
            new_model.layers.append(layer.clone())

    def parse(self, nn_filename):
        if nn_filename.endswith(".h5"):
            layers, n_layers, input_shape = KerasParser.parse_file(nn_filename)
            self.layers = layers
            self.n_layers = n_layers
            self.input_shape = input_shape
        # elif nn_filename.endswith(".nnet"):
        #         layers, n_layers, input_shape = NnetParser.parse_file(nn_filename)
        #         self.layers = layers
        #         self.n_layers = n_layers
        #         self.input_shape = input_shape
        else:
            raise Exception("Unsupported network model file format", nn_filename)

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)

        return outputs