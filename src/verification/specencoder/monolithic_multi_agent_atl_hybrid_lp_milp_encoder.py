from src.utils.utils import get_widest_bounds
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.specencoder.monolithic_boolean_milp_encoder import MonolithicBooleanMILPEncoder


class MonolithicMultiAgentATLHybridLinearMixedIntegerEncoder(MonolithicBooleanMILPEncoder):
    def __init__(self, constrs_manager, state_vars, gamma, not_gamma, env):
        """
        An immutable visitor implementation for constructing a set of MILPs from a formula.
        :param constrs_manager: Constraints manager.
        :param state_vars: The current state of the env.
        :param agent: The neural agent.
        :param env: The non-deterministic environment.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        super(MonolithicMultiAgentATLHybridLinearMixedIntegerEncoder, self).__init__(constrs_manager, state_vars)

        self.bf = env.get_branching_factor()
        self.env = env
        self.next_vars = self.state_vars

        self.gamma = gamma
        self.not_gamma = not_gamma
        self.num_joint_actions_not_gamma = self.bf ** len(self.not_gamma)  # If {}, BF^0 = 1 => only nop action enabled.
        self.num_joint_actions_gamma = self.bf ** len(self.gamma)

        self.constrs_to_add = []

    # *** Implementation of previous version of encoding. ***
    def visitGammaExistentialFormula(self, element):
        """
        Existential formula visitor.
        :param element:
        :return:
        """

        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import GammaExistentialFormula
            smaller_formula = GammaExistentialFormula(k - 1, element.gamma, element.not_gamma, element.left)

        binvars = set()
        constrs_to_add = []

        # Allocate binary variables for BF^{|Agt|} possible next states.
        delta = self.constrs_manager.create_binary_variables(self.num_joint_actions_gamma)
        constrs_to_add.append(self.constrs_manager.get_sum_constraint(delta, 1))

        self.constrs_manager.update()

        init_vars = self.state_vars

        gamma_see_vars = []
        not_gamma_see_vars = []
        for agent in element.gamma:
            see_vars, see_constrs = agent.get_constraints_for_obs(self.constrs_manager, init_vars)
            constrs_to_add.extend(see_constrs)
            gamma_see_vars.append(see_vars)
            self.constrs_manager.get_variable_tracker().add_action_variables(see_vars)

        for agent in element.not_gamma:
            see_vars, see_constrs = agent.get_constraints_for_obs(self.constrs_manager, init_vars)
            constrs_to_add.extend(see_constrs)
            not_gamma_see_vars.append(see_vars)
            self.constrs_manager.get_variable_tracker().add_action_variables(see_vars)

        # succ_vars are vars corresponding to y_j. There are BF^|¬Gamma| sets of them. Needed when
        # computing leaf constraints recursively.
        succ_vars = []

        # Worst case bounds represented as tuples corresponding to each y_j.
        succ_vars_bounds = []
        for joint_action_not_gamma_idx in range(self.num_joint_actions_not_gamma):
            succ_vars.append(self.constrs_manager.create_state_variables(len(init_vars)))
            succ_vars_bounds.append([(float("inf"), float("-inf")) for _ in range(len(init_vars))])

        for joint_action_gamma_idx in range(self.num_joint_actions_gamma):
            for joint_action_not_gamma_idx in range(self.num_joint_actions_not_gamma):
                # Add and get constraints for transition function.

                # y_ij = output_state_vars
                output_state_vars, output_state_constrs, _ \
                    = self.env.get_constraints_for_transition(
                    self.constrs_manager, gamma_see_vars, not_gamma_see_vars, init_vars)

                # Compute max possible upper and min possible lower bounds for each output var.
                get_widest_bounds(succ_vars_bounds[joint_action_not_gamma_idx], output_state_vars)

                # Compute constraints { y_j == y_ij }
                next_var_constrs = [nsv == osv for nsv, osv in zip(succ_vars[joint_action_not_gamma_idx], output_state_vars)]

                action_constrs, _ = \
                    self.env.get_constraints_for_joint_protocol(
                        self.constrs_manager, self.state_vars, joint_action_gamma_idx, joint_action_not_gamma_idx,
                        gamma_see_vars, not_gamma_see_vars, output_state_vars, binvars)

                for constr in output_state_constrs + next_var_constrs + action_constrs:
                    if constr._sense != 'I':  # Hack to check if indicator constraint.
                        constrs_to_add.append(self.constrs_manager
                                              .create_indicator_constraint(delta[joint_action_gamma_idx], 1, constr))
                    else:
                        constrs_to_add.append(constr)

        for joint_action_not_gamma_idx in range(self.num_joint_actions_not_gamma):
            # Add computed bounds for the y_j's in constr manager.
            output_lower, output_upper = zip(*succ_vars_bounds[joint_action_not_gamma_idx])  # Unzip the bounds.
            self.constrs_manager.add_variable_bounds(succ_vars[joint_action_not_gamma_idx], HyperRectangleBounds(output_lower, output_upper))

            self.state_vars = succ_vars[joint_action_not_gamma_idx]  # y_j

            # Get constrs for phi(y_j)
            left_constraints = smaller_formula.acceptI(self)
            constrs_to_add.extend(left_constraints)

        self.state_vars = init_vars  # reset to x

        self.constrs_manager.binvars.update(binvars)

        return constrs_to_add

    def visitGammaUniversalFormula(self, element):
        """
        Universal formula visitor.

        [[Gamma]] phi

        :param element:
        :return:
        """
        gamma = element.gamma
        not_gamma = element.not_gamma
        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import GammaUniversalFormula
            smaller_formula = GammaUniversalFormula(k - 1, element.gamma, element.not_gamma, element.left)

        binvars = set()
        constrs_to_add = []

        # Allocate matrix of binary variables.
        deltas = []
        for _ in range(self.num_joint_actions_gamma):
            deltas.append(self.constrs_manager.create_binary_variables(self.num_joint_actions_not_gamma))
        # Note we don't add sum constraints yet for deltas. Add them in outer loop.

        self.constrs_manager.update()

        init_vars = self.state_vars

        gamma_see_vars = []
        not_gamma_see_vars = []

        for agent in gamma:
            see_vars, see_constrs = agent.get_constraints_for_obs(self.constrs_manager, init_vars)
            constrs_to_add.extend(see_constrs)
            gamma_see_vars.append(see_vars)

        for agent in not_gamma:
            see_vars, see_constrs = agent.get_constraints_for_obs(self.constrs_manager, init_vars)
            constrs_to_add.extend(see_constrs)
            not_gamma_see_vars.append(see_vars)

        self.constrs_manager.get_variable_tracker().add_action_variables([gamma_see_vars, not_gamma_see_vars])

        # Here we need succ_vars to correspond to the y_i's, instead of y_j's.
        succ_vars = []
        succ_vars_bounds = []

        # We eventually need bounds on the y_i's.
        for _ in range(self.num_joint_actions_gamma):
            succ_vars.append(self.constrs_manager.create_state_variables(len(init_vars)))
            succ_vars_bounds.append([(float("inf"), float("-inf")) for _ in range(len(init_vars))])

        for joint_action_gamma_idx in range(self.num_joint_actions_gamma):
            # Add the sum constrains for all deltas first.
            constrs_to_add.append(self.constrs_manager.get_sum_constraint(deltas[joint_action_gamma_idx], 1))

            for joint_action_not_gamma_idx in range(self.num_joint_actions_not_gamma):

                # y_ij = output_state_vars
                output_state_vars, output_state_constrs = \
                    self.env.get_constraints_for_transition(
                        self.constrs_manager, gamma_see_vars, not_gamma_see_vars, init_vars)

                # Compute max possible upper and min possible lower bounds for each output var.
                get_widest_bounds(succ_vars_bounds[joint_action_gamma_idx], output_state_vars)

                action_constrs, _ = \
                    self.env.get_constraints_for_joint_protocol(
                        self.constrs_manager, self.state_vars, joint_action_gamma_idx, joint_action_not_gamma_idx,
                        gamma_see_vars, not_gamma_see_vars, output_state_vars)

                # Compute constraints { y_i == y_ij }.
                # Gut feeling is that there is a lot of duplicated constraints here.
                next_var_constrs = [nsv == osv for nsv, osv in
                                    zip(succ_vars[joint_action_gamma_idx], output_state_vars)]

                # Note, we do not add the next_var_constrs here, because they are not to the RHS of the implication.
                for constr in output_state_constrs + action_constrs + next_var_constrs:
                    if constr._sense != 'I':  # Hack to check if indicator constraint.
                        constrs_to_add.append(
                            self.constrs_manager.create_indicator_constraint(
                                deltas[joint_action_gamma_idx][joint_action_not_gamma_idx], 1, constr))
                    else:
                        constrs_to_add.append(constr)

        for joint_action_gamma_idx in range(self.num_joint_actions_gamma):
            # Is this correct? In any case, add bounds for the y_i's to the constrmanager.
            output_lower, output_upper = zip(*succ_vars_bounds[joint_action_gamma_idx])  # Unzip the bounds.
            self.constrs_manager.add_variable_bounds(succ_vars[joint_action_gamma_idx],
                                                     HyperRectangleBounds(output_lower, output_upper))

            self.state_vars = succ_vars[joint_action_gamma_idx]  # y_i

            # Get constrs for phi(y_i)
            left_constraints = smaller_formula.acceptI(self)
            constrs_to_add.extend(left_constraints)

        self.constrs_manager.get_variable_tracker().add_state_variables(succ_vars)

        self.constrs_manager.binvars.update(binvars)

        return constrs_to_add
