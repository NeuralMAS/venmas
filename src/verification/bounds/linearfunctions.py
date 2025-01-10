from src.utils.utils import *


class LinearFunctions:
    """
    matrix is an (n x m) np array
    offset is an (n) np array

    An object represents n linear functions f(i) of m input variables x

    f(i) = matrix[i]*x + offset[i]

    """
    def __init__(self, matrix, offset):
        self.size = matrix.shape[0]
        self.matrix = matrix
        self.offset = offset

    def clone(self):
        return LinearFunctions(self.matrix.copy(), self.offset.copy())

    def get_size(self):
        return self.size

    def get_matrix(self):
        return self.matrix

    def get_offset(self):
        return self.offset

    def compute_max_values(self, input_bounds):
        return get_positive_part(self.matrix).dot(input_bounds.get_upper()) + \
               get_negative_part(self.matrix).dot(input_bounds.get_lower()) + \
               self.offset

    def compute_min_values(self, input_bounds):
        return get_positive_part(self.matrix).dot(input_bounds.get_lower()) + \
               get_negative_part(self.matrix).dot(input_bounds.get_upper()) + \
               self.offset
