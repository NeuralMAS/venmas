import itertools

from src.verification.specencoder.breadth_first_compositional_boolean_milp_encoder import \
    CompositionalBooleanMILPEncoder


class BreadthFirstCompositionalMultiAgentATLMILPEncoder(CompositionalBooleanMILPEncoder):
    def __init__(self, constrs_manager, state_vars, gamma, not_gamma, env):
        """
        An immutable visitor implementation for constructing a set of MILPs from an ATL formula.
        :param constrs_manager: Constraints manager.
        :param state_vars: The current state of the env.
        :param agent: The neural agent.
        :param env: The non-deterministic environment.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        super(BreadthFirstCompositionalMultiAgentATLMILPEncoder, self).__init__(constrs_manager, state_vars)

        self.bf = env.get_branching_factor()
        self.env = env
        self.next_vars = self.state_vars

        self.gamma = gamma
        self.not_gamma = not_gamma
        self.num_joint_actions_not_gamma = self.bf ** len(self.not_gamma)  # If {}, BF^0 = 1 => only nop action enabled.
        self.num_joint_actions_gamma = self.bf ** len(self.gamma)

    def visitGammaExistentialFormula(self, element):

        # Add constraints for agent network.
        gamma_obs_vars = []
        not_gamma_obs_vars = []

        gamma_obs_constrs = []
        not_gamma_obs_constrs = []
        for agent in element.gamma:
            see_vars, see_constrs = agent.get_constraints_for_obs(self.constrs_manager, self.state_vars)
            gamma_obs_constrs.append(see_constrs)
            gamma_obs_vars.append(see_vars)
            self.constrs_manager.get_variable_tracker().add_action_variables(see_vars)

        for agent in element.not_gamma:
            see_vars, see_constrs = agent.get_constraints_for_obs(self.constrs_manager, self.state_vars)
            not_gamma_obs_constrs.append(see_constrs)
            not_gamma_obs_vars.append(see_vars)
            self.constrs_manager.get_variable_tracker().add_action_variables(see_vars)

        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import GammaExistentialFormula
            smaller_formula = GammaExistentialFormula(k - 1, element.gamma, element.not_gamma, element.left)

        root_state_vars = self.state_vars
        joint_protocol_constrs = []
        binvars = set()

        for joint_action_gamma_idx in range(self.num_joint_actions_gamma):
            full_transition_milps = []

            for joint_action_not_gamma_idx in range(self.num_joint_actions_not_gamma):
                output_state_vars, output_state_constrs, _ \
                    = self.env.get_constraints_for_transition(
                    self.constrs_manager, gamma_obs_vars, not_gamma_obs_vars, self.state_vars)

                # Add constraints for joint protocol (joint_action_gamma_idx, joint_action_not_gamma_idx).
                action_constrs, _ = self.env.get_constraints_for_joint_protocol(
                    self.constrs_manager, self.state_vars, joint_action_gamma_idx, joint_action_not_gamma_idx,
                    gamma_obs_vars, not_gamma_obs_vars, output_state_vars, binvars)

                current_transition_milp = [action_constrs + output_state_constrs]

                # Only used in recursive step. Need to reset to root for remaining loops.
                self.state_vars = output_state_vars
                smaller_formula_milps = smaller_formula.acceptI(self)

                full_transition_milp = self.local_cartesian_product(current_transition_milp, smaller_formula_milps)
                full_transition_milps.append(full_transition_milp)

                self.state_vars = root_state_vars  # Remember the root.

            # Get global cartesian product of all milps generated for all transitions of not_gamma.
            joint_protocol_constr = self.global_pseudo_cartesian_product(full_transition_milps)
            joint_protocol_constrs.append(joint_protocol_constr)

        # Flatten obs constrs into a single list.
        global_obs_constrs = self.flatten_list_of_lists_to_list(gamma_obs_constrs + not_gamma_obs_constrs)

        # Gamma observed and not_gamma observed, for every possible transition via each joint protocol app.
        # These constraints are a list of pairs, namely:
        global_transition_constrs = [self.local_cartesian_product([global_obs_constrs], c)
                                     for c in joint_protocol_constrs]

        # Above is now a list, where each element is a list of MILPs each containing observation constraint.
        # Now need to flatten into list.

        existential_global_transition_constrs = self.flatten_list_of_lists_to_list(global_transition_constrs)

        return existential_global_transition_constrs

    def flatten_list_of_lists_to_list(self, l):
        return [elem for elems in l for elem in elems]

    # Get a cartesian product between two milps.
    def local_cartesian_product(self, milp1, milp2):
        return list(l + r for l, r in itertools.product(milp1, milp2))

    # Get a "global" cartesian product of a list of milps.
    def global_pseudo_cartesian_product(self, milps):
        return list(self.flatten_list_of_lists_to_list(l) for l in itertools.product(*milps))

    def visitGammaUniversalFormula(self, element):
        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import GammaUniversalFormula
            smaller_formula = GammaUniversalFormula(k - 1, element.gamma, element.not_gamma, element.left)

        binvars = set()

        root_state_vars = self.state_vars

        gamma_obs_vars = []
        not_gamma_obs_vars = []

        gamma_obs_constrs = []
        not_gamma_obs_constrs = []

        local_products = []

        for agent in element.gamma:
            see_vars, see_constrs = agent.get_constraints_for_obs(self.constrs_manager, root_state_vars)
            gamma_obs_constrs.append(see_constrs)
            gamma_obs_vars.append(see_vars)

        for agent in element.not_gamma:
            see_vars, see_constrs = agent.get_constraints_for_obs(self.constrs_manager, root_state_vars)
            not_gamma_obs_constrs.append(see_constrs)
            not_gamma_obs_vars.append(see_vars)

        self.constrs_manager.get_variable_tracker().add_action_variables(gamma_obs_vars + not_gamma_obs_vars)

        succ_vars = []
        for joint_action_gamma_idx in range(self.num_joint_actions_gamma):  # i
            current_transition_milps = []
            output_state_vars, output_state_constrs = self.env.get_constraints_for_transition(
                self.constrs_manager, gamma_obs_vars, not_gamma_obs_vars, root_state_vars)

            succ_vars.append(output_state_vars)

            for joint_action_not_gamma_idx in range(self.num_joint_actions_not_gamma):  # j
                # Add and get constraints for joint protocol function.
                action_constrs, _ = self.env.get_constraints_for_joint_protocol(
                        self.constrs_manager, root_state_vars, joint_action_gamma_idx, joint_action_not_gamma_idx,
                        gamma_obs_vars, not_gamma_obs_vars, output_state_vars, binvars)

                # List of constraints for global observation.
                flattened_obs_constrs = self.flatten_list_of_lists_to_list(gamma_obs_constrs + not_gamma_obs_constrs)

                # Singleton MILP composed of constraints for the current transition.
                current_transition_milp = flattened_obs_constrs + action_constrs + output_state_constrs
                current_transition_milps.append(current_transition_milp)

            # Setup state vars for recursion.
            self.state_vars = output_state_vars

            # Recurse.
            smaller_formula_milps = smaller_formula.acceptI(self)

            # Construct MILPs to check if formula holds in the ith Gamma joint action for some j not(Gamma) joint action.
            local_product = self.local_cartesian_product(current_transition_milps, smaller_formula_milps)
            local_products.append(local_product)

            # Remember the root.
            self.state_vars = root_state_vars

        self.constrs_manager.get_variable_tracker().add_state_variables(succ_vars)

        # Do a global product of the list of local products.
        result = self.global_pseudo_cartesian_product(local_products)
        return result
