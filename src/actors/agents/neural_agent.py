from src.actors.agents.agent import Agent
from src.network_parser.network_model import NetworkModel
from src.utils.constants import NeuralOutputSettings


class NeuralAgent(Agent):
    """
    Class representing an agent whose protocol function is defined exactly by a neural network.
    Gets as a parameter an instance of NetworkModel, our internal representation of a network
    """
    def __init__(self, network_model, output_setting=NeuralOutputSettings.INTEGER_ARGMAX):
        """
        :param network_model: an instance of NetworkModel,
            an internal representation of a neural network
        :param output_setting: a value indicating whether the actual output
            of the agent's protocol function is simply the tuple of output q_values,
            a one-hot or an integer representation of the argmax of the output layer.
            Default value is integer representation.
        """
        assert isinstance(network_model, NetworkModel)

        super(NeuralAgent, self).__init__()

        self.network_model = network_model
        self.output_setting = output_setting

    def get_constraints_for_action(self, constrs_manager, input_state_vars):
        return self.get_network_constraints(self.network_model, self.output_setting, constrs_manager, input_state_vars)

    @staticmethod
    def get_network_constraints(network_model, output_setting, constrs_manager, input_state_vars):
        assert isinstance(network_model, NetworkModel)

        constrs = []

        # Get the constraints for the network. Get list of q values.
        action_q_vars, network_constrs = constrs_manager.get_network_constraints(network_model.layers, input_state_vars)
        constrs.extend(network_constrs)

        if output_setting == NeuralOutputSettings.ONE_HOT_ARGMAX or \
                output_setting == NeuralOutputSettings.INTEGER_ARGMAX:
            # Encode argmax from the above q-values.
            argmax_vars, argmax_constrs = constrs_manager.get_argmax_constraints(action_q_vars)
            constrs.extend(argmax_constrs)

            if output_setting == NeuralOutputSettings.INTEGER_ARGMAX:
                # Return a single integer variable as the output of argmax.
                integer_argmax_var, integer_argmax_constrs = constrs_manager.get_argmax_index_constraints(argmax_vars)
                constrs.extend(integer_argmax_constrs)

                action_vars = [integer_argmax_var]
            else:
                action_vars = argmax_vars
        else:
            # Otherwise return the q-values
            action_vars = action_q_vars

        return action_vars, constrs
