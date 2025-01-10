from abc import ABCMeta, abstractmethod

from src.utils.constants import NOT_IMPLEMENTED
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.complete.constrmanager.gurobi_constraints_manager import GurobiConstraintsManager
from src.verification.complete.constrmanager.custom_constraints_manager import CustomConstraintsManager


class MultiAgentEnvironment:
    __metaclass__ = ABCMeta

    """
    An interface for an environment, with deterministic 
    transition function. Such transition function is modelled
    as piecewise linear functions.
    """

    @abstractmethod
    def get_constraints_for_initial_state(self, constraints_manager, input_bounds):
        """
        Produces variables for the initial state with appropriate bounds.
        :param constraints_manager: Gurobi constraints representing the current
        state/situation
        :param input_bounds: HyperRectangleBounds representing the input set constraints
        :side-effects: May modify constrs_manager.
        :return: Variables for the initial state.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_constraints_for_transition(
            self, constraints_manager, joint_action_vars_gamma, joint_action_vars_not_gamma, state_vars):
        """
        Return the constraints encoding the transition function of the environment
        with output variables representing the next state to be input to the agent.
        :param constraints_manager: Constraints manager for managing the current
        set of constraints, variables, etc.
        :param joint_action_vars_gamma: List of Lists of Gurobi variables representing joint action of agents in gamma.
        :param joint_action_vars_not_gamma: List of Lists of Gurobi variables representing joint action of agents in
        not(gamma).
        :param state_vars: Variables representing the state of the environment.
        :side-effects: May modify constrs_manager.
        :return: The variables representing the state after the transition,
        the constraints and the set of binary variables used in indicator constraints.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_constraints_for_joint_protocol(
            self, constraints_manager, state_vars, joint_action_gamma_idx, joint_action_not_gamma_idx, gamma_see_vars,
            not_gamma_see_vars, output_state_vars):
        """
        Return the constraints encoding an application of a (global) joint protocol function being executed by all
        agents in Agt.
        :param constraints_manager:
        :param state_vars:
        :param joint_action_gamma_idx: Integer representing a joint action contained in X_{i \in Gamma} prot_i(q)
        :param joint_action_not_gamma_idx: Integer representing a joint action contained in X_{j \not\in Gamma} prot_j(q)
               if |not(Gamma)| = 0, then joint_action_not_gamma_idx = 0 only.
        :param gamma_see_vars:  List of Lists of Gurobi variables representing "joint observation" of agents in Gamma.
        :param not_gamma_see_vars: List of Lists of Gurobi variables representing "joint observation" of agents in not(Gamma).
        :param output_state_vars: List of Gurobi variables allocated for the next state of the environment.
        :return: The variables representing the state after application of te joint protocol,
        the constraints and the set of binary variables used in indicator constraints.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_branching_factor(self):
        """
        Returns the branching factor.
        :return: The branching factor.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_indices_gamma(self, idx):
        """
        Returns a tuple of indices representing a joint action, given an index into a set of all possible joint actions.
        :return: The tuple.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_indices_not_gamma(self, idx):
        """
        Returns a tuple of indices representing a joint action, given an index into a set of all possible joint actions.
        :return: The tuple.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)


class AbstractMultiAgentEnvironment(MultiAgentEnvironment):
    __metaclass__ = ABCMeta

    """
    An abstract class for an environment implementing two methods, 
    get_constraints_for_initial_state and get_branching_factor,
    these methods do not depend on how the environment transition function 
    is actually given, whether via a neural network or a set of linear constraints.
    """

    def __init__(self, branching_factor):
        self.branching_factor = branching_factor

    def get_constraints_for_initial_state(self, constraints_manager, input_bounds):
        """
        Produces variables with appropriate bounds for the initial state.
        Depending of constraints manager, the variables can be Gurobi variables or internal variables.
        """

        # Currently support only hyper-rectangle input bounds
        if not isinstance(input_bounds, HyperRectangleBounds):
            raise Exception("Currently only HyperRectangleBounds are accepted as input bounds. Instead got",
                            input_bounds)

        # variables for the initial state
        state_vars = constraints_manager.create_state_variables(input_bounds.get_size(),
                                                                lbs=input_bounds.get_lower(),
                                                                ubs=input_bounds.get_upper())

        # Add the initial state variables to constr_manager's tracker.
        # Useful for getting counter-examples
        constraints_manager.get_variable_tracker().add_state_variables(state_vars)

        if isinstance(constraints_manager, GurobiConstraintsManager) \
                or isinstance(constraints_manager, CustomConstraintsManager):
            # Record the bounds of the variables.
            # Useful for bounds computation/propagation
            constraints_manager.add_variable_bounds(state_vars, input_bounds)

        return state_vars, []

    def get_branching_factor(self):
        """
        Returns the branching factor.
        :return: The branching factor.
        """
        return self.branching_factor

