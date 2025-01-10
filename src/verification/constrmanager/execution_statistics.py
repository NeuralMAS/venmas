class ExecutionStatistics:
    def __init__(self, number_of_calls, max_var_number, max_constr_number):
        self.number_of_calls = number_of_calls
        self.max_var_number = max_var_number
        self.max_constr_number = max_constr_number
        self.witness_states = []
        self.witness_actions = []
        self.active_number = 0
        self.inactive_number = 0
        self.incompatible_number = 0
        self.at_least_one_number = 0
        self.implications_number = 0

    def update(self, var_number, constr_number):
        self.number_of_calls += 1
        if var_number > self.max_var_number:
            self.max_var_number = var_number
        if constr_number > self.max_constr_number:
            self.max_constr_number = constr_number

    def set_witnesses(self, witness_states, witness_actions):
        self.witness_states = witness_states
        self.witness_actions = witness_actions

    def add_cc_stats(self, active, inactive, incompatible, at_least_one, implications):
        self.active_number = sum([len(set) for set in active])
        self.inactive_number = sum([len(set) for set in inactive])
        self.incompatible_number = sum([len(set) for set in incompatible])
        self.at_least_one_number = sum([len(set) for set in at_least_one])
        self.implications_number = sum([len(set) for set in implications])