from src.actors.envs.environment import Environment
from src.network_parser.network_model import NetworkModel


class NonDeterministicNeuralEnvironment(Environment):
    """
    Class representing an environment whose transition function is
    non-deterministic and is defined exactly by a number of neural networks.
    Gets as a parameter a list of NetworkModels, our internal representation of a network.
    """

    def __init__(self, network_models):
        """
        :param network_models: a list of instances of NetworkModel,
            an internal representation of a neural network
        """
        for model in network_models:
            assert isinstance(model, NetworkModel)

        super(NonDeterministicNeuralEnvironment, self).__init__(len(network_models))

        self.network_models = network_models

    def get_constraints_for_transition(
            self, i, constraints_manager, action_vars, input_state_vars):

        assert 0 <= i < self.branching_factor

        # Get the output state variables
        # and the constraints for the network itself
        output_state_vars, constrs = constraints_manager.get_network_constraints(self.network_models[i].layers,
                                                                                 input_state_vars + action_vars)

        return output_state_vars, constrs

    def get_branching_factor_opt(self, state_vars, action_vars):
        return self.branching_factor

