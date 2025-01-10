#!/usr/bin/env python
import sys
sys.path.append('../../../../')

import datetime
from timeit import default_timer as timer
from gurobipy import Model

from src.verification.specencoder.monolithic_single_agent_ctl_milp_encoder import \
    MonolithicSingleAgentCTLMILPEncoder
from src.verification.specencoder.monolithic_multi_agent_ctl_milp_encoder import \
    MonolithicMultiAgentCTLMILPEncoder
from src.verification.specencoder.depth_first_compositional_single_agent_ex_milp_encoder import \
    DepthFirstCompositionalSingleAgentExistsNextMILPEncoder
from src.verification.specencoder.depth_first_compositional_multi_agent_ex_milp_encoder import \
    DepthFirstCompositionalMultiAgentExistsNextMILPEncoder
from src.verification.specencoder.breadth_first_compositional_multi_agent_atl_milp_encoder import \
    BreadthFirstCompositionalMultiAgentATLMILPEncoder
from src.verification.specencoder.monolithic_multi_agent_atl_hybrid_lp_milp_encoder import \
    MonolithicMultiAgentATLHybridLinearMixedIntegerEncoder


from src.verification.constrmanager.gurobi_constraints_manager import GurobiConstraintsManager
from src.verification.constrmanager.custom_constraints_manager import CustomConstraintsManager
from src.verification.multiprocessing.aesverifier import AESVerifier
from src.utils.formula_visitors.immutable_nnf_visitor import FormulaVisitorNNF
from src.utils.formula import *


"""
Useful functions for verifying temporal properties of agent-environment systems (single or multiple agents)
"""

TO_USER_RESULT = {"True": "False", "False": "True", "Timeout": "Timeout", "Interrupted": "Interrupted"}


def mono_single_ctl_verify(formula, input_hyper_rectangle, agent, env, timeout=3600):
    # print("Monolithic encoding")

    # struct to store solving stats
    log_info = []

    start = timer()
    print("Formula ", formula)

    print("Start: ", datetime.datetime.now())  # Do not delete

    # Create the Gurobi constraints manager to get a single program
    gmodel = Model("AES")
    gmodel.Params.LogToConsole = 0
    gmodel.Params.TimeLimit = timeout

    constraint_manager = GurobiConstraintsManager(gmodel)

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraint_manager, input_hyper_rectangle)
    mono_visitor = MonolithicSingleAgentCTLMILPEncoder(constraint_manager, initial_state_vars, agent, env)

    # Compute the set of MILP constraints for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    milp_constrs = negated_formula.acceptI(mono_visitor)

    # add the constraints and check feasibility of the resulting encoding
    constraint_manager.add_constrs(milp_constrs)
    result = constraint_manager.check_feasibility()

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Negate the result
    result = TO_USER_RESULT[result]

    # store results and total time
    log_info.append(runtime)
    log_info.append(result)

    print("Overall result and time:", result, runtime)
    stats = constraint_manager.stats
    print("Max number of variables  ", stats.max_var_number)
    print("Max number of constraints", stats.max_constr_number)

    if result == "False":
        print("Counter-example:")
        depth = len(stats.witness_states)
        for i in range(0, depth - 1):
            print("\t", "state", i, ":", stats.witness_states[i])
            print("\t", "action", i, ":", stats.witness_actions[i])
        print("\t", "state", depth - 1, ":", stats.witness_states[depth - 1])

    return log_info


def seq_single_ctl_verify(formula, input_hyper_rectangle, agent, env, timeout=3600, workers_n=1):
    """
    Verify specification using compositional encoding in sequential manner.

    :param env: A non-deterministic environment.
    :param agent: A neural agent.
    :param formula: Temporal specification to verify.
    :param input_hyper_rectangle: Input bounds.
    :return: None
    """

    print("Sequential Compositional encoding - {} workers".format(workers_n))
    log_info = []

    start = timer()
    print("Formula ", formula)
    print("Start: ", datetime.datetime.now())  # Do not delete

    # Create the custom constraints manager to get a number of (small) programs
    constraint_manager = CustomConstraintsManager()

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraint_manager, input_hyper_rectangle)
    # ctlverifier_visitor = FormulaMultiMILPBuilderVisitor(constraint_manager, initial_state_vars, agent, env)
    ## For the experiments, we use the more efficient encoder that only supports formulas
    ## with EX, disjunction and conjunction of atomic formulas
    ctlverifier_visitor = DepthFirstCompositionalSingleAgentExistsNextMILPEncoder(constraint_manager, initial_state_vars, agent, env)

    # Create a pool verifier for the MILP builder visitor and for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    aesverifier = AESVerifier(ctlverifier_visitor, negated_formula, workers_n)

    AESVerifier.TIME_LIMIT = timeout
    AESVerifier.PARALLEL_PROCESSES_NUMBER = workers_n

    result, job_id, extra = aesverifier.verify()

    # Negate the result
    result = TO_USER_RESULT[result]

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Store results and total time
    log_info.append(runtime)
    log_info.append(result)

    print("Overall result and time:", result, runtime, "job n", job_id)
    if result == "False":
        print("\t\tCounter-example:")
        print("\n".join(["{}".format(item) for item in extra]))
    print("")

    return log_info


def parallel_single_ctl_verify(formula, input_hyper_rectangle, agent, env, timeout=3600, workers_n=8):
    """
    Verify specification using compositional encoding in parallel manner.

    :param env: A non-deterministic environment.
    :param agent: A neural agent.
    :param formula: Temporal specification to verify.
    :param input_hyper_rectangle: Input bounds.
    :return: None
    """

    print("Parallel Compositional encoding - {} workers".format(workers_n))
    log_info = []

    start = timer()
    print("Formula ", formula)
    print("Start: ", datetime.datetime.now())  # Do not delete

    # Create the custom constraints manager to get a number of (small) programs
    constraint_manager = CustomConstraintsManager()

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraint_manager, input_hyper_rectangle)
    # ctlverifier_visitor = BreadthFirstCompositionalSingleAgentCTLMILPEncoder(constraint_manager, initial_state_vars, agent, env)
    ## For the experiments, we use the more efficient encoder that only supports formulas
    ## with EX, disjunction and conjunction of atomic formulas
    ctlverifier_visitor = DepthFirstCompositionalSingleAgentExistsNextMILPEncoder(constraint_manager, initial_state_vars, agent, env)

    # Create a pool verifier for the MILP builder visitor and for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    aesverifier = AESVerifier(ctlverifier_visitor, negated_formula, workers_n)

    AESVerifier.TIME_LIMIT = timeout
    AESVerifier.PARALLEL_PROCESSES_NUMBER = workers_n

    result, job_id, extra = aesverifier.verify()

    # Negate the result
    result = TO_USER_RESULT[result]

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Store results and total time
    log_info.append(runtime)
    log_info.append(result)

    print("Overall result and time:", result, runtime, "job n", job_id)
    if result == "False":
        print("\t\tCounter-example:")
        print("\n".join(["{}".format(item) for item in extra]))
    print("")

    return log_info


# Parallel-poly (--method 0)
def parallel_multi_atl_verify(formula, input_hyper_rectangle, gamma, not_gamma, env, timeout):
    """
    Verify a multi-agent neural system using polylithic approach using parallel execution.

    :param formula: An ATL formula
    :param input_hyper_rectangle: Hyperrectangle representing initial state.
    :param gamma: Group of agents in coalition.
    :param not_gamma: Group of agents in complement of gamma.
    :param env: Multi-agent environment.
    :param timeout: Timeout in minutes.
    :return: Void.
    """
    # Create the custom constraints manager to get a number of (small) programs
    constraints_manager = CustomConstraintsManager()

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraints_manager, input_hyper_rectangle)
    atlverifier_encoder = BreadthFirstCompositionalMultiAgentATLMILPEncoder(constraints_manager, initial_state_vars,
                                                                            gamma, not_gamma, env)

    # Create a pool verifier for the MILP encoder and for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    aesverifier = AESVerifier(atlverifier_encoder, negated_formula, 4)

    AESVerifier.TIME_LIMIT = timeout

    start = timer()
    print("Formula ", formula)
    print("Start: ", datetime.datetime.now())  # Do not delete
    result, job_id, extra = aesverifier.verify()
    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Negate the result
    result = TO_USER_RESULT[result]

    print("Overall result and time:", result, runtime, "job n", job_id)
    if result == "False":
        print("\t\tCounter-example:")
        print(extra)
    print("")


# Sequential-poly (--method 1)
def sequential_multi_atl_verify(formula, input_hyper_rectangle, gamma, not_gamma, env, timeout):
    """
    Verify using the polylithic approach with sequential execution.
    :param formula: An ATL formula
    :param input_hyper_rectangle: Hyperrectangle representing initial state.
    :param gamma: Group of agents in coalition.
    :param not_gamma: Group of agents in complement of gamma.
    :param env: Multi-agent environment.
    :param timeout: Timeout in minutes.
    :return: Void.
    """
    # Create the custom constraints manager to get a number of (small) programs
    constraints_manager = CustomConstraintsManager()

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraints_manager, input_hyper_rectangle)
    atlverifier_encoder = BreadthFirstCompositionalMultiAgentATLMILPEncoder(constraints_manager, initial_state_vars,
                                                                            gamma, not_gamma, env)

    # Create a pool verifier for the MILP encoder and for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    aesverifier = AESVerifier(atlverifier_encoder, negated_formula, 1)

    AESVerifier.TIME_LIMIT = timeout
    AESVerifier.PARALLEL_PROCESSES_NUMBER = 1

    start = timer()
    print("Formula ", formula)
    print("Start: ", datetime.datetime.now())  # Do not delete
    result, job_id, extra = aesverifier.verify()
    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Negate the result
    result = TO_USER_RESULT[result]

    print("Overall result and time:", result, runtime, "job n", job_id)
    if result == "False":
        print("\t\tCounter-example:")
        print(extra)
    print("")


def mono_multi_atl_verify(formula, input_hyper_rectangle, gamma, not_gamma, env, timeout):
    """
    Verify using the monolithic approach.

    :param formula: An ATL formula
    :param input_hyper_rectangle: Hyperrectangle representing initial state.
    :param gamma: Group of agents in coalition.
    :param not_gamma: Group of agents in complement of gamma.
    :param env: Multi-agent environment.
    :param timeout: Timeout in minutes.

    :return: Void.
    """
    start = timer()
    print("Formula ", formula)

    print("Start: ", datetime.datetime.now())  # Do not delete

    # Create the Gurobi constraints manager to get a single program
    gmodel = Model("AES")
    gmodel.Params.LogToConsole = 0
    gmodel.Params.TimeLimit = timeout

    constraint_manager = GurobiConstraintsManager(gmodel)

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraint_manager, input_hyper_rectangle)
    mono_visitor = MonolithicMultiAgentATLHybridLinearMixedIntegerEncoder(constraint_manager, initial_state_vars, gamma,
                                                                          not_gamma, env)

    # Compute the set of MILP constraints for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    milp_constrs = negated_formula.acceptI(mono_visitor)

    # add the constraints and check feasibility of the resulting encoding
    constraint_manager.add_constrs(milp_constrs)
    result = constraint_manager.check_feasibility()

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Negate the result
    result = TO_USER_RESULT[result]

    print("Overall result and time:", result, runtime)
    stats = constraint_manager.stats
    print("Max number of variables  ", stats.max_var_number)
    print("Max number of constraints", stats.max_constr_number)

    if result == "False":
        print("Counter-example:")
        depth = len(stats.witness_states)
        for i in range(0, depth - 1):
            print("\t", "state", i, ":", stats.witness_states[i])
            print("\t", "action", i, ":", stats.witness_actions[i])
        print("\t", "state", depth - 1, ":", stats.witness_states[depth - 1])


# Mono (--method 2)
def mono_multi_atl_hybrid_verify(formula, input_hyper_rectangle, gamma, not_gamma, env, timeout):
    """
    Verify using the monolithic over-approximation (hybrid: mixing LP and MILP) approach.

    :param formula: An ATL formula
    :param input_hyper_rectangle: Hyperrectangle representing initial state.
    :param gamma: Group of agents in coalition.
    :param not_gamma: Group of agents in complement of gamma.
    :param env: Multi-agent environment.
    :param timeout: Timeout in minutes.

    :return: Void.
    """
    start = timer()
    print("Formula ", formula)

    print("Start: ", datetime.datetime.now())  # Do not delete

    # Create the Gurobi constraints manager to get a single program
    gmodel = Model("AES")
    gmodel.Params.LogToConsole = 0
    gmodel.Params.TimeLimit = timeout

    constraint_manager = GurobiConstraintsManager(gmodel)

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraint_manager, input_hyper_rectangle)
    mono_visitor = MonolithicMultiAgentATLHybridLinearMixedIntegerEncoder(constraint_manager, initial_state_vars, gamma,
                                                                          not_gamma, env)

    # Compute the set of MILP constraints for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    milp_constrs = negated_formula.acceptI(mono_visitor)

    # add the constraints and check feasibility of the resulting encoding
    constraint_manager.add_constrs(milp_constrs)
    result = constraint_manager.check_feasibility()

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Negate the result
    result = TO_USER_RESULT[result]

    print("Overall result and time:", result, runtime)
    stats = constraint_manager.stats
    print("Max number of variables  ", stats.max_var_number)
    print("Max number of constraints", stats.max_constr_number)

    if result == "False":
        print("Counter-example:")
        depth = len(stats.witness_states)
        for i in range(0, depth - 1):
            print("\t", "state", i, ":", stats.witness_states[i])
            print("\t", "action", i, ":", stats.witness_actions[i])
        print("\t", "state", depth - 1, ":", stats.witness_states[depth - 1])


def mono_multi_ctl_verify(formula, input_hyper_rectangle, agents, env, timeout):
    """
    Verify using the monolithic approach.
    :param formula: A CTL formula
    :param input_hyper_rectangle: Hyperrectangle representing initial state.
    :param gamma: Group of agents in coalition.
    :param not_gamma: Group of agents in complement of gamma.
    :param env: Multi-agent environment.
    :param timeout: Timeout in minutes.

    :return: Void.
    """
    start = timer()
    print("Formula ", formula)

    print("Start: ", datetime.datetime.now())  # Do not delete

    # Create the Gurobi constraints manager to get a single program
    gmodel = Model("AES")
    gmodel.Params.LogToConsole = 0
    gmodel.Params.TimeLimit = timeout

    constraint_manager = GurobiConstraintsManager(gmodel)

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraint_manager, input_hyper_rectangle)
    mono_visitor = MonolithicMultiAgentCTLMILPEncoder(constraint_manager, initial_state_vars, agents, env)

    invert_result = True
    # If the specification is a reachability specification, we pass it as it is
    if isinstance(formula, ENextFormula):
        milp_constrs = formula.acceptI(mono_visitor)
        # no need to invert the result in this case
        invert_result = False
    # Otherwise we negate the formula
    elif isinstance(formula, ANextFormula):
        # Compute the set of MILP constraints for the negation of the formula in NNF
        negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
        milp_constrs = negated_formula.acceptI(mono_visitor)
    else:
        milp_constrs = []

    # add the constraints and check feasibility of the resulting encoding
    constraint_manager.add_constrs(milp_constrs)
    result = constraint_manager.check_feasibility()

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Negate the result
    user_result = TO_USER_RESULT[result] if invert_result else result

    print("Overall result and time:", user_result, runtime)
    stats = constraint_manager.stats
    print("Max number of variables  ", stats.max_var_number)
    print("Max number of constraints", stats.max_constr_number)

    if result == "True":
        print("Counter-example:")
        depth = len(stats.witness_states)
        for i in range(0, depth - 1):
            print("\t", "state", i, ":", [round(item) for item in stats.witness_states[i]])
            print("\t", "action", i, ":", [round(item) for item in stats.witness_actions[i]])
        print("\t", "state", depth - 1, ":", [round(item) for item in stats.witness_states[depth - 1]])


def parallel_multi_exists_next_verify(formula, input_hyper_rectangle, agents, env, timeout):
    """
    Verify a multi-agent neural system using polylithic approach using parallel execution.

    :param formula: An (EX,AND,OR) or a (AX,ANDd,OR) formula
    :param input_hyper_rectangle: Hyperrectangle representing initial state.
    :param agents: Group of agents.
    :param env: Multi-agent environment.
    :param timeout: Timeout in minutes.
    :return: Void.
    """
    # Create the custom constraints manager to get a number of (small) programs
    constraints_manager = CustomConstraintsManager()

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraints_manager, input_hyper_rectangle)
    atlverifier_encoder = DepthFirstCompositionalMultiAgentExistsNextMILPEncoder(constraints_manager, initial_state_vars, agents, env)

    if isinstance(formula, ENextFormula):
        aesverifier = AESVerifier(atlverifier_encoder, formula, 4)
    elif isinstance(formula, ANextFormula):
        # Create a pool verifier for the MILP encoder and for the negation of the formula in NNF
        negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
        aesverifier = AESVerifier(atlverifier_encoder, negated_formula, 4)
    else:
        aesverifier = None

    AESVerifier.TIME_LIMIT = timeout

    start = timer()
    print("Formula ", formula)
    print("Start: ", datetime.datetime.now())  # Do not delete
    result, job_id, extra = aesverifier.verify()
    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Negate the result
    result = TO_USER_RESULT[result]

    print("Overall result and time:", result, runtime, "job n", job_id)
    if result == "False":
        print("\t\tCounter-example:")
        print(extra)
    print("")
