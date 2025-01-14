from src.actors.envs.environment import Environment
from src.verification.bounds.bounds import HyperRectangleBounds


class GuardingEnv(Environment):

    def __init__(self):
        """
        GuardingEnv implements the environment for the Guarding Game.
        It is a dummy environment.

        The state is the local environment state as per the paper
        """
        # This ia a dummy environment.
        # The branching factor is 1, and the state is always 0
        super(GuardingEnv, self).__init__(1)

        # Number of dimensions of the local env state;
        self.STATE_SPACE_DIM = 1

    def get_constraints_for_transition(self, constrs_manager, input_state_vars, env_action_vars, agent_joint_action_vars):
        """
        Get constraints for a global transition.
        :param constrs_manager:
        :param input_state_vars:
        :param env_action_vars:
        :param agent_joint_action_vars:
        :return:
        """

        # A list used to contain Gurobi constraints to be added to the constraints manager.
        constrs_to_add = []

        # Compute bounds for the next state.
        input_variable_bounds = constrs_manager.get_variable_bounds(input_state_vars)
        input_variable_bounds_lower = input_variable_bounds.get_lower()
        input_variable_bounds_upper = input_variable_bounds.get_upper()

        # Create next state variables with the bounds provided.
        output_state_vars = constrs_manager.create_state_variables(len(input_state_vars),
                                                   lbs=input_variable_bounds_lower, ubs=input_variable_bounds_upper)
        constrs_manager.add_variable_bounds(output_state_vars,
                                            HyperRectangleBounds(input_variable_bounds_lower, input_variable_bounds_upper))

        for var1, var2 in zip(input_state_vars, output_state_vars):
            constrs_to_add.append(
                constrs_manager.get_equality_constraint(var1, var2))

        return output_state_vars, constrs_to_add

    def get_constraints_for_protocol(self, constrs_manager, input_state_vars):
        """A dummy protocol"""
        action = constrs_manager.create_integer_variable(lb=0, ub=0)

        return [[action]], [[]]
