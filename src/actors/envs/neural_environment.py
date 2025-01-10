from src.actors.envs.environment import Environment
from src.network_parser.network_model import NetworkModel


class NeuralEnvironment(Environment):
    """
    Class representing an environment whose transition function is
    deterministic and is defined exactly by a neural network.
    Gets as a parameter an instance of NetworkModel, our internal representation of a network.
    """

    def __init__(self, network_model):
        """
        :param network_model: an instance of NetworkModel,
            an internal representation of a neural network
        """
        assert isinstance(network_model, NetworkModel)
        super(NeuralEnvironment, self).__init__(1)

        self.network_model = network_model

    def get_constraints_for_transition(
            self, i, constraints_manager, action_vars, input_state_vars):

        # Get the output state variables
        # and the constraints for the network itself
        output_state_vars, constrs = constraints_manager.get_network_constraints(self.network_model.layers,
                                                                                 input_state_vars + action_vars)

        return output_state_vars, constrs

