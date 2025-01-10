import numpy as np


def get_widest_bounds(output_bounds, state_vars):
    for j, var in enumerate(state_vars):
        l, u = output_bounds[j]
        output_bounds[j] = (min(var.lb, l), max(var.ub, u))


def get_positive_part(weights):
    return np.maximum(weights, np.zeros(weights.shape))


def get_negative_part(weights):
    return np.minimum(weights, np.zeros(weights.shape))


def compute_lower(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_lower) + weights_minus.dot(input_upper)


def compute_upper(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_upper) + weights_minus.dot(input_lower)


def compute_lin_lower_and_upper(weights_minus, weights_plus, bias, lower_matrix, upper_matrix,
                                lower_offset, upper_offset):

    return compute_lower(weights_minus, weights_plus, lower_matrix, upper_matrix), \
           compute_lower(weights_minus, weights_plus, lower_offset, upper_offset) + bias, \
           compute_upper(weights_minus, weights_plus, lower_matrix, upper_matrix), \
           compute_upper(weights_minus, weights_plus, lower_offset, upper_offset) + bias


def increment_indices(indices, max_ranges):
    """
    Increments indices similarly to a binary counter only
    that every index is bounded by its maximum range.

    Returns True when the maximum has been reached for all indices.
    Otherwise returns False
    :param indices: a list of integer indiced to increment
    :param max_ranges: a list of maximum ranges for each of the indices
    :return:
        whether the maximum has been reached
    """
    # Start from the rightmost index
    i = len(indices) - 1

    # Increment the indices
    while i >= 0:
        indices[i] += 1

        # Check if the current index exceeds its maximum range
        if indices[i] >= max_ranges[i]:
            # Reset the current index to 0 and move to the left
            indices[i] = 0
            i -= 1
        else:
            # If the current index is within its maximum range, we're done
            break

    # If i becomes negative, all indices have reached their maximum, reset them all
    if i < 0:
        return True

    return False
