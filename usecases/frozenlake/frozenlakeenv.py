import numpy as np

from src.actors.envs.environment import Environment
from src.utils.utils import compute_lower, get_positive_part, get_negative_part
from src.verification.bounds.bounds import HyperRectangleBounds
from operator import __ge__, __eq__


class FrozenLakeEnv(Environment):
    """
    The Frozen lake environment for the 3 x 3 grid world.
    """

    DIRECTION_TRANSFORMATIONS = np.array([
        # the chosen direction unchanged
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        # to the left of the chosen direction
        [[0, 0, 0, 1],
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0]],
        # to the right of the chosen direction
        [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, 0, 0, 0]]
    ])

    DIRECTION_X = np.array([-1, 0, 1, 0])
    DIRECTION_Y = np.array([0, 1, 0, -1])

    STATE_X = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]) # the column
    STATE_Y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]) # the row

    MIN_2D = 1
    MAX_2D = 3

    INDEX_TO_TWO_DIMENSIONAL = np.array([
        [1,1], [1,2], [1,3],
        [2,1], [2,2], [2,3],
        [3,1], [3,2], [3,3]
    ])

    def __init__(self):
        """
        FrozenLakeEnv represents the environment of the FrozenLake scenario.
        Given a chosen direction by the agent, the environment can move the agent
        in the given direction, or to the left, or to the right of that direction.
        :param state_space_dim: The dimension of the *environment* state space.
        """
        super(FrozenLakeEnv, self).__init__(3)

        # Constants.
        # 3 x 3 world map
        #  F(1) F(2) H(3)
        #  F(4) F(5) F(6)
        #  H(7) F(8) G(9)
        self.STATE_SPACE_DIM = 9

        # There are 4 possible actions, left (1), down (2), right (3), up (4),
        # encoded as one-hot vectors
        self.ACTION_SPACE_DIM = 4

    def get_branching_factor_opt(self, input_state_vars, action_vars):
        return self.branching_factor

    def get_constraints_for_transition(self, i, constrs_manager, action_vars, input_state_vars):
        """
        Add the constraints for the transition function itself, create constraints to determine the
        Gurobi variables representing the next state to be input back into the agent. Constraints
        are only added to the Gurobi model by the caller, to reduce side-effects in this function.
        :param i: The transition function used to move to the next state.
        :param constrs_manager: Gurobi constraints representing the current state/situation.
        :param action_vars: Gurobi variables representing the agent's action.
        :param input_state_vars: Gurobi variables representing the state of the **scenario**.
        :side-effects: Modifies constrs_manager when adding variables.
        :return: The Gurobi variables representing the state after the transition has been made.
        """

        assert len(action_vars) == self.ACTION_SPACE_DIM
        assert len(input_state_vars) == self.STATE_SPACE_DIM

        # A list used to contain Gurobi constraints to be added to the constraints manager.
        constrs_to_add = []

        # Compute direction variables, coinciding with the action variables for i = 0,
        # and transformed for i=1 or i=2 (to the left of to the right of the original direction)
        action_bounds = constrs_manager.get_variable_bounds(action_vars)
        direction_lower = action_bounds.get_lower()
        direction_upper = action_bounds.get_upper()
        if i != 0:
            direction_lower = self.DIRECTION_TRANSFORMATIONS[i].dot(direction_lower)
            direction_upper = self.DIRECTION_TRANSFORMATIONS[i].dot(direction_upper)
        direction_vars = constrs_manager.create_state_variables(self.ACTION_SPACE_DIM, lbs=direction_lower, ubs=direction_upper)
        constrs_to_add.extend([
            constrs_manager.get_linear_constraint(action_vars + [direction_vars[r]], list(self.DIRECTION_TRANSFORMATIONS[i][r]) + [-1], 0)
            for r in range(self.ACTION_SPACE_DIM)
        ])

        # Transform the direction variables (that are one-hot vectors of length 4)
        # to 2-dimensional vectors
        # (-1,0) for left, (0,1) for down, (1,0) for right and (0,-1) for up
        dir_x_lower = self.get_lower_bounds(self.DIRECTION_X, direction_lower, direction_upper)
        dir_x_upper = self.get_upper_bounds(self.DIRECTION_X, direction_lower, direction_upper)
        dir_y_lower = self.get_lower_bounds(self.DIRECTION_Y, direction_lower, direction_upper)
        dir_y_upper = self.get_upper_bounds(self.DIRECTION_Y, direction_lower, direction_upper)
        [dir_x, dir_y] = constrs_manager.create_state_variables(2,
                                                                lbs=[dir_x_lower, dir_y_lower],
                                                                ubs=[dir_x_upper, dir_y_upper])
        constrs_to_add.extend([
            constrs_manager.get_linear_constraint(direction_vars + [dir_x], list(self.DIRECTION_X) + [-1], 0),
            constrs_manager.get_linear_constraint(direction_vars + [dir_y], list(self.DIRECTION_Y) + [-1], 0)
        ])

        # Transform the state variables (that are one-hot vectors of length 9)
        # to 2-dimensional vectors
        # (1,1) (1,2) (1,3)
        # (2,1) (2,2) (2,3)
        # (3,1) (3,2) (3,3)
        input_bounds = constrs_manager.get_variable_bounds(input_state_vars)
        state_x_lower = self.get_lower_bounds(self.STATE_X, input_bounds.get_lower(), input_bounds.get_upper())
        state_x_upper = self.get_upper_bounds(self.STATE_X, input_bounds.get_lower(), input_bounds.get_upper())
        state_y_lower = self.get_lower_bounds(self.STATE_Y, input_bounds.get_lower(), input_bounds.get_upper())
        state_y_upper = self.get_upper_bounds(self.STATE_Y, input_bounds.get_lower(), input_bounds.get_upper())
        [state_x, state_y] = constrs_manager.create_state_variables(2,
                                                                    lbs=[state_x_lower, state_y_lower],
                                                                    ubs=[state_x_upper, state_y_upper])
        constrs_to_add.extend([
            constrs_manager.get_linear_constraint(input_state_vars + [state_x], list(self.STATE_X) + [-1], 0),
            constrs_manager.get_linear_constraint(input_state_vars + [state_y], list(self.STATE_Y) + [-1], 0)
        ])

        # Compute the next 2-dimensional state variables
        next_state_pre_x_lower = state_x_lower + dir_x_lower
        next_state_pre_x_upper = state_x_upper + dir_x_upper
        next_state_pre_y_lower = state_y_lower + dir_y_lower
        next_state_pre_y_upper = state_y_upper + dir_y_upper
        next_state_x_lower = min(max(next_state_pre_x_lower, self.MIN_2D), self.MAX_2D)
        next_state_x_upper = min(max(next_state_pre_x_upper, self.MIN_2D), self.MAX_2D)
        next_state_y_lower = min(max(next_state_pre_y_lower, self.MIN_2D), self.MAX_2D)
        next_state_y_upper = min(max(next_state_pre_y_upper, self.MIN_2D), self.MAX_2D)

        [next_state_pre_x, next_state_x, next_state_pre_y, next_state_y] = \
            constrs_manager.create_state_variables(
                4,
                lbs=[next_state_pre_x_lower, next_state_x_lower, next_state_pre_y_lower, next_state_y_lower],
                ubs=[next_state_pre_x_upper, next_state_x_upper, next_state_pre_y_upper, next_state_y_upper])

        constrs_to_add.extend(
            [
                constrs_manager.get_linear_constraint([state_x, dir_x, next_state_pre_x], [1, 1, -1], 0),
                constrs_manager.get_linear_constraint([state_y, dir_y, next_state_pre_y], [1, 1, -1], 0)
            ] +
            self.get_min_max_constraints(constrs_manager, next_state_pre_x, next_state_x) +
            self.get_min_max_constraints(constrs_manager, next_state_pre_y, next_state_y)
        )

        # Convert the 2-dimensional state variables to the one-hot encoding
        min_next_state = int((next_state_y_lower - 1) * self.MAX_2D + next_state_x_lower)
        max_next_state = int((next_state_y_upper - 1) * self.MAX_2D + next_state_x_upper)
        next_state_lower = [0 for _ in range(self.STATE_SPACE_DIM)]
        next_state_upper = [0 for _ in range(self.STATE_SPACE_DIM)]
        for state_n in range(min_next_state - 1, max_next_state):
            next_state_upper[state_n] = 1
        if min_next_state == max_next_state:
            next_state_lower[min_next_state-1] = 1

        next_state_vars = constrs_manager.create_state_variables(self.STATE_SPACE_DIM, lbs=next_state_lower, ubs=next_state_upper)
        constrs_manager.add_variable_bounds(next_state_vars,
                                            HyperRectangleBounds(next_state_lower, next_state_upper))

        # Binary variables for distinguishing between all possible states
        delta = constrs_manager.create_binary_variables(self.STATE_SPACE_DIM)
        constrs_to_add.append(constrs_manager.get_sum_constraint(delta, 1))

        constrs_to_add.extend([constrs_manager.get_assignment_constraint(delta[state_n], 0)
                               for state_n in range(min_next_state-1)])
        constrs_to_add.extend([constrs_manager.get_assignment_constraint(delta[state_n], 0)
                               for state_n in range(max_next_state, self.STATE_SPACE_DIM)])
        for state_n in range(min_next_state-1, max_next_state):
            # Detect what state we are in and set next_state_vars accordingly
            constrs = \
                [constrs_manager.get_assignment_constraint(next_state_x, self.INDEX_TO_TWO_DIMENSIONAL[state_n][0]),
                 constrs_manager.get_assignment_constraint(next_state_y, self.INDEX_TO_TWO_DIMENSIONAL[state_n][1]),
                 constrs_manager.get_assignment_constraint(next_state_vars[state_n], 1)] \
                + \
                [constrs_manager.get_assignment_constraint(next_state_vars[i], 0)
                 for i in range(self.STATE_SPACE_DIM) if not i == state_n]

            constrs_to_add.extend([
                constrs_manager.create_indicator_constraint(delta[state_n], 1, constr) for constr in constrs])

        return next_state_vars, constrs_to_add

    def get_min_max_constraints(self, constrs_manager, next_state_pre_var, next_state_var):
        """
        Constraints for handling extreme cases,
        e.g., when the input state is a left-most state and the direction is left.

        :param constrs_manager:
        :param next_state_pre_var:
        :param next_state_var:
        :return:
        """
        [min, max, middle] = constrs_manager.create_binary_variables(3)

        return [
            constrs_manager.get_sum_constraint([min, max, middle], 1),
            constrs_manager.create_indicator_constraint(
                min, 1,
                constrs_manager.get_assignment_constraint(next_state_pre_var, self.MIN_2D - 1)),
            constrs_manager.create_indicator_constraint(
                min, 1,
                constrs_manager.get_assignment_constraint(next_state_var, self.MIN_2D)),
            constrs_manager.create_indicator_constraint(
                max, 1,
                constrs_manager.get_assignment_constraint(next_state_pre_var, self.MAX_2D + 1)),
            constrs_manager.create_indicator_constraint(
                max, 1,
                constrs_manager.get_assignment_constraint(next_state_var, self.MAX_2D)),
            constrs_manager.create_indicator_constraint(
                middle, 1,
                constrs_manager.get_ge_constraint(next_state_pre_var, self.MIN_2D)),
            constrs_manager.create_indicator_constraint(
                middle, 1,
                constrs_manager.get_le_constraint(next_state_pre_var, self.MAX_2D)),
            constrs_manager.create_indicator_constraint(
                middle, 1,
                constrs_manager.get_equality_constraint(next_state_var, next_state_pre_var))
        ]

    def get_lower_bounds(self, coeffs, lower, upper):
        return get_positive_part(coeffs).dot(lower) + \
               get_negative_part(coeffs).dot(upper)

    def get_upper_bounds(self, coeffs, lower, upper):
        return get_positive_part(coeffs).dot(upper) + \
               get_negative_part(coeffs).dot(lower)

    # @staticmethod
    # def compute_next_state_bounds(input_variable_bounds, acc_bounds):
    #     input_variable_bounds_lower = input_variable_bounds.get_lower()
    #     input_variable_bounds_upper = input_variable_bounds.get_upper()
    #     acc_lower, acc_upper = acc_bounds.get_dimension_bounds(0)
    #
    #     next_state_lower = \
    #         [input_variable_bounds_lower[VcasConstants.INTRUDER_ALTITUDE] -
    #          VcasConstants.DELTA_T * input_variable_bounds_upper[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE] -
    #          VcasConstants.DELTA_T**2 * 0.5 * acc_upper,
    #          input_variable_bounds_lower[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE] + VcasConstants.DELTA_T * acc_lower,
    #          input_variable_bounds_lower[VcasConstants.TAU] - VcasConstants.DELTA_T,
    #         ]
    #
    #     next_state_upper = \
    #         [input_variable_bounds_upper[VcasConstants.INTRUDER_ALTITUDE] -
    #          VcasConstants.DELTA_T * input_variable_bounds_lower[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE] -
    #          VcasConstants.DELTA_T**2 * 0.5 * acc_lower,
    #          input_variable_bounds_upper[VcasConstants.OWNSHIP_VERTICAL_CLIMBRATE] + acc_upper,
    #          input_variable_bounds_upper[VcasConstants.TAU] - VcasConstants.DELTA_T
    #         ]
    #     return next_state_lower, next_state_upper
