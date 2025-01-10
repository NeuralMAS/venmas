from abc import ABCMeta, abstractmethod

from src.utils.constants import NOT_IMPLEMENTED


class Agent:
    """ Agent interface.

    Responsible for computing the agent's next action given a state in the
    environment.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_constraints_for_action(self, constrs_manager, input_state_vars):
        """
        Get the constraints for the network itself, set constraints to
        determine the action taken by the agent. Will need to compute the
        action of the agent based on the Q-value with highest value.
        :param constrs_manager: Constraint manager handling all constraints.
        :param input_state_vars: Set of variables representing the state that
        is input to the agent's protocol function.
        :side-effects: May modify constrs_manager.
        :return:
            variables representing action taken,
            the constraints and
            the binary variables used in indicator constraints.
        """
        return NotImplementedError(NOT_IMPLEMENTED)


class MultiAgent:
    """ Multi-Agent interface.

    Responsible for computing the observation function of the agent.
    Returns the percept.

    The implementation for KR 2020/Jaamas 2022
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_constraints_for_obs(self, constrs_manager, input_state_vars):
        """
        Create Gurobi constraints for performing an observation on the env.
        Get the constraints for the network itself, set constraints to
        determine the percept output by the agent. Will need to compute the
        percept of the agent based on the Q-value with highest value.
        :param constrs_manager: Constraint manager handling all constraints.
        :param input_state_vars: Set of variables representing the state that
        is input to the agent's observation function.
        :side-effects: May modify constrs_manager.
        :return:
            variables representing the percept,
            the constraints to compute it
        """
        return NotImplementedError(NOT_IMPLEMENTED)


class TemplateAgent(MultiAgent):
    """ TemplateAgent interface.

    Responsible for computing the agent's observation, protocol and transition functions.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_constraints_for_obs(self, constrs_manager, input_state_vars):
        """
        Create Gurobi constraints for performing an observation on the env.
        Get the constraints for the network itself, set constraints to
        determine the percept output by the agent. Will need to compute the
        percept of the agent based on the Q-value with highest value.
        :param constrs_manager: Constraint manager handling all constraints.
        :param input_state_vars: Set of variables representing the state that
        is input to the agent's observation function.
        :side-effects: May modify constrs_manager.
        :return:
            variables representing the percept,
            the constraints to compute it
        """
        return NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_constraints_for_protocol(self, constrs_manager, input_state_vars):
        """
        Create constraints for computing the available actions in the given state.
        :param constrs_manager: Constraint manager handling all constraints.
        :param input_state_vars: Set of variables representing the state that
        is input to the agent's protocol function.
        :side-effects: May modify constrs_manager.
        :return:
            a list of action vars and a list of action constraints,
            where i-th element corresponds to i-th possible action.
        """
        return NotImplementedError(NOT_IMPLEMENTED)

    def get_constraints_for_transition(self, constrs_manager, input_state_vars,
                                       own_action_vars, joint_action_vars, env_action_vars):
        """
        Create Gurobi constraints for performing a transition, given the action of the agent,
        the set of actions by all the agents and the action of the env.

        :param constrs_manager: Constraint manager handling all constraints.
        :param input_state_vars: Set of variables representing the state that
        is input to the agent's observation function.
        :param own_action_vars: Set of variables representing the action taken by the agent
        :param joint_action_vars: Set of variables representing the set of actions taken by all agents
        :param env_action_vars: Set of variables representing the action taken by the environment
        :side-effects: May modify constrs_manager.
        :return:
            variables representing the new state,
            the constraints to compute it
        """
        return NotImplementedError(NOT_IMPLEMENTED)

    def get_branching_factor(self):
        """
        Returns the maximum number of actions the agent can take
        """
        return NotImplementedError(NOT_IMPLEMENTED)