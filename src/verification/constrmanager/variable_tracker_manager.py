from src.verification.complete.constrmanager.aes_variable_tracker import AESVariableTracker


class VariableTrackerManager:
    def __init__(self):
        self.aes_var_tracker = AESVariableTracker()

    def add_state_variables_to_tracker(self, state_vars):
        self.aes_var_tracker.add_state_variables(state_vars)

    def add_action_variables_to_tracker(self, action_vars):
        self.aes_var_tracker.add_action_variables(action_vars)

    def add_q_variables_to_tracker(self, q_vars):
        self.aes_var_tracker.add_q_variables(q_vars)

