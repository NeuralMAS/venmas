from abc import ABCMeta, abstractmethod

from src.utils.constants import NOT_IMPLEMENTED
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.constrmanager.gurobi_constraints_manager import GurobiConstraintsManager
from src.verification.constrmanager.custom_constraints_manager import CustomConstraintsManager


class EnvironmentI:
    __metaclass__ = ABCMeta

    """
    An interface for an environment, possibly with non-deterministic 
    transition function. Such transition function is modelled
    as k piecewise linear functions, and k is referred to as
    the branching factor of the transition function.
    """

    @abstractmethod
    def get_constraints_for_initial_state(self, constraints_manager, input_bounds):
        """
        Produces variables for the initial state with appropriate bounds.
        :param constraints_manager: Gurobi constraints representing the current
        state/situation
        :param input_bounds: HyperRectangleBounds representing the input set constraints
        :side-effects: May modify constrs_manager.
        :return: Variables for the initial state, and
        the constraints encoding the set of initial states.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_constraints_for_transition(
            self, i, constraints_manager, action_vars, state_vars):
        """
        Return the constraints encoding the transition function of the environment
        with output variables representing the next state to be input to the agent.
        :param i: The transition function used to move to the next state.
        :param constraints_manager: Constraints manager for managing the current
        set of constraints, variables, etc.
        :param action_vars: Variables representing the agent's action.
        :param state_vars: Variables representing the state of the environment.
        :side-effects: May modify constrs_manager.
        :return: The variables representing the state after the transition,
        the constraints and the set of binary variables used in indicator constraints.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_branching_factor(self):
        """
        Returns the branching factor of the environment transition function.
        :return: The branching factor.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)


class Environment(EnvironmentI):
    __metaclass__ = ABCMeta

    """
    An environment class implementing two methods, 
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

        if isinstance(constraints_manager, GurobiConstraintsManager) or \
                isinstance(constraints_manager, CustomConstraintsManager):
            # Record the bounds of the variables.
            # Useful for bounds computation/propagation
            constraints_manager.add_variable_bounds(state_vars, input_bounds)

        # We do not need explicit constraints for describing a hyper-rectangle
        return state_vars, []

    def get_branching_factor(self):
        """
        Returns the branching factor of the environment transition function.
        :return: The branching factor.
        """
        return self.branching_factor

