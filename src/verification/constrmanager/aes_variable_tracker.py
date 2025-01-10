from collections.abc import Iterable

from gurobipy import Var

from src.utils.variable_interface import Variable


class AESVariableTracker:
    def __init__(self):
        self.path_state_vars = []
        self.path_action_vars = []
        self.path_q_vars = []

    def add_state_variables(self, state_vars):
        self.path_state_vars.append(state_vars)

    def pop_state_variables(self):
        self.path_state_vars.pop()

    def add_action_variables(self, action_vars):
        self.path_action_vars.append(action_vars)

    def pop_action_variables(self):
        self.path_action_vars.pop()

    def add_q_variables(self, q_vars):
        self.path_q_vars.append(q_vars)

    def get_witness_states(self, model):
        return [
            self.get_assignment(layer, model)
            for layer in self.path_state_vars
        ]

    def get_witness_actions(self, model):
        return [
            self.get_assignment(layer, model)
            for layer in self.path_action_vars
        ]

    def get_assignment(self, entity, model):
        if isinstance(entity, Iterable):
            return [self.get_assignment(item, model) for item in entity]

        if isinstance(entity, Variable):
            return model.getVarByName(entity.varName).x

        if isinstance(entity, Var):
            return entity.x

    def get_trace(self):
        return [self.get_vars(layer) for layer in self.path_state_vars], \
               [self.get_vars(layer) for layer in self.path_action_vars]

    @staticmethod
    def get_variable_values(model, variables):
        return [
            [model.getVarByName(var.varName).x for var in layer]
            for layer in variables
        ]

    def get_vars(self, entity):
        if isinstance(entity, Iterable):
            flattened = []
            for item in entity:
                vars = self.get_vars(item)
                if isinstance(vars, Iterable):
                    flattened.extend(vars)
                else:
                    flattened.append(vars)
            return flattened

        if isinstance(entity, Variable) or isinstance(entity, Var):
            return entity
