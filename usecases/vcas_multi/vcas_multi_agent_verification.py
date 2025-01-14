#!/usr/bin/env python

import argparse
import collections
import sys
sys.path.append('../../../')

import os

from usecases.vcas.gamma_intruder_two_agent_vcasenv import TwoAgentIntruderVcasEnv
from usecases.vcas.gamma_ownship_two_agent_vcasenv import TwoAgentOwnshipVcasEnv

from usecases.vcas.two_agent_vcas_agents import VcasOwnshipAgent, VcasIntruderAgent
from usecases.vcas.gamma_ownship_intruder_two_agent_vcasenv import VcasConstants, TwoAgentOwnshipIntruderVcasEnv

from src.network_parser.network_model import NetworkModel
from src.utils.formula import *
from src.verification.bounds.bounds import HyperRectangleBounds

from src.verification.utils.verification_utils import parallel_multi_atl_verify, sequential_multi_atl_verify, mono_multi_atl_verify, \
    mono_multi_atl_hybrid_verify


def main():
    parser = argparse.ArgumentParser(description="Verify a MANS")
    parser.add_argument("-m", "--method", type=int, default=2, help="Method to use for verification: 0. Parallel-poly; 1. Sequential-poly; 2. Mono; 3. Mono-hybrid. ")
    parser.add_argument("-f", "--formula", type=int, default=1, help="Formula to verify: 0. [[o,i]] X^k safety; 1. <<o>> X^k safety; 2. [[i]] X^k safety; 3. <<o,i>> X^k unsafety")
    parser.add_argument("--initial_pos", default=-50.0, type=float, help="Initial position of pilot, in [pos - noise, pos + noise].")
    parser.add_argument("-n", "--noise", default=2.0, type=float, help="Noise to add to initial position of pilot.")
    parser.add_argument("-h0own", "--initial_climbrate_own", default=-5.0, type=float, help="Initial climbrate of ownship.")
    parser.add_argument("-h0int", "--initial_climbrate_int", default=5.0, type=float, help="Initial climbrate of intruder.")
    parser.add_argument("-t", "--initial_tau", default=25.0, type=float, help="Initial tau.")
    parser.add_argument("-aown", "--initial_acceleration_own", default=VcasConstants.G / 10, type=float, help="Initial acceleration of ownship.")
    parser.add_argument("-aint", "--initial_acceleration_int", default=-VcasConstants.G / 10, type=float, help="Initial acceleration of intruder.")
    parser.add_argument("-k", "--max_steps", default=2, type=int, help="Maximum number of time steps to verify for.")
    parser.add_argument("-l", "--max_steps_inner", default=1, type=int, help="Number of time steps to verify for inner EX in AXEX.")
    parser.add_argument("-k2", "--step", default=1, type=int, help="One single time step to verify for.")
    parser.add_argument("-l2", "--step_inner", default=1, type=int, help="One single time to to verify for inner EX in AXEX.")
    parser.add_argument("--single", action="store_true", help="Whether should run for a single timestep or multiple time steps")
    parser.add_argument("-w", "--workers", default=2, type=int, help="Number of workers.")
    parser.add_argument("-to", "--timeout", default=3600, type=int, help="Timeout in minutes.")

    ARGS = parser.parse_args()
    # ** Note ** , when adding a new formula, add appropriate agent config to list.
    agent_configs = [2, 0, 1, 2, 1]
    agents, env = initialise_and_get_agent_and_env(agent_configs[ARGS.formula])

    # Constraint specific variables of the initial state to one value by setting the upper
    # bounds equal to the lower bounds.
    initial_advisory_int = VcasConstants.COC
    initial_advisory_own = VcasConstants.COC

    unzipped = zip(*[(ARGS.initial_pos - ARGS.noise,  ARGS.initial_pos + ARGS.noise),
                     (ARGS.initial_climbrate_own,     ARGS.initial_climbrate_own),
                     (ARGS.initial_climbrate_int,     ARGS.initial_climbrate_int),
                     (ARGS.initial_tau,               ARGS.initial_tau),
                     (ARGS.initial_acceleration_own,  ARGS.initial_acceleration_own),
                     (ARGS.initial_acceleration_int,  ARGS.initial_acceleration_int),
                     (initial_advisory_own,           initial_advisory_own),
                     (initial_advisory_int,           initial_advisory_int)])

    input_hyper_rectangle = HyperRectangleBounds(*unzipped)
    print(input_hyper_rectangle)

    steps = [ARGS.step] if ARGS.single else range(1, ARGS.max_steps + 1)

    for num_steps in steps:
        print(num_steps, "steps")

        gamma, not_gamma, formula = get_formula_and_gamma(ARGS, agents, num_steps)

        verification_methods = [parallel_multi_atl_verify, sequential_multi_atl_verify, mono_multi_atl_verify, mono_multi_atl_hybrid_verify]

        print("Formula to verify", formula)
        # Run a method.
        verification_methods[ARGS.method](formula, input_hyper_rectangle, gamma, not_gamma, env, ARGS.timeout)
        print("\n")


def get_formula_and_gamma(ARGS, agents, num_steps):
    unsafe = AtomicConjFormula(VarConstConstraint(StateCoordinate(VcasConstants.INT_ALT_IDX), LT, 100),
                               VarConstConstraint(StateCoordinate(VcasConstants.INT_ALT_IDX), GT, -100))

    safe = AtomicDisjFormula(VarConstConstraint(StateCoordinate(VcasConstants.INT_ALT_IDX), GT, 100),
                             VarConstConstraint(StateCoordinate(VcasConstants.INT_ALT_IDX), LT, -100))
    gamma_both = agents
    not_gamma_empty = []

    gamma_ownship = [agents[0]]
    not_gamma_intruder = [agents[1]]

    gamma_intruder = [agents[1]]
    not_gamma_ownship = [agents[0]]

    safety_formula_gamma_both = GammaUniversalFormula(num_steps, gamma_both, not_gamma_empty, safe) # [[o,i]] X^k safety
    safety_formula_gamma_intruder = GammaUniversalFormula(num_steps, gamma_intruder, not_gamma_ownship, safe) # [[o]] X^k safety
    existential_safety_formula_gamma_ownship = GammaExistentialFormula(num_steps, gamma_ownship, not_gamma_intruder, safe) # <<o>> X^k safe
    unsafety_formula = GammaExistentialFormula(num_steps, gamma_both, not_gamma_empty, unsafe) # <<o,i>> X^k nmac
    unsafety_formula_int = GammaExistentialFormula(num_steps, gamma_intruder, not_gamma_ownship, unsafe) # <<i>> X^k nmac

    FormulaInfo = collections.namedtuple('FormulaInfo', 'formula, gamma, not_gamma')
    FORMULA_INFO = [
        FormulaInfo(safety_formula_gamma_both, gamma_both, not_gamma_empty),
        FormulaInfo(existential_safety_formula_gamma_ownship, gamma_ownship, not_gamma_intruder),
        FormulaInfo(safety_formula_gamma_intruder, gamma_intruder, not_gamma_ownship),
        FormulaInfo(unsafety_formula, gamma_both, not_gamma_empty),
        FormulaInfo(unsafety_formula_int, gamma_intruder, not_gamma_ownship),
    ]

    # response_formula = ANextFormula(num_steps, DisjFormula(safe, ENextFormula(ARGS.max_steps_inner, safe)))

    chosen_formula = FORMULA_INFO[ARGS.formula].formula
    chosen_gamma = FORMULA_INFO[ARGS.formula].gamma
    chosen_not_gamma = FORMULA_INFO[ARGS.formula].not_gamma

    return chosen_gamma, chosen_not_gamma, chosen_formula


def initialise_and_get_agent_and_env(agent_config):
    """
    Initialise agent and environment.
    :return: List of initialised VcasAgent object and a VcasEnv object.
    """

    # Where the agent networks are.
    REL_PATH = "../resources/vcas_single-v0/vcas_{}.h5"

    # Dimension of the action space; there is only one action, the advisory, returned by the agent.
    ACTION_SPACE_DIM = 1

    # Dimension of the state space;
    # This is len([position,
    #              vertical_climbrate_own,
    #              vertical_climbrate_int,
    #              tau, prev_adv_own,
    #              prev_adv_int,
    #              acceleration_own,
    #              acceleration_int]
    #             ) = 8.
    STATE_SPACE_DIM = 8

    network_models = []
    for i in range(VcasConstants.N_ADVISORIES):
        nmodel = NetworkModel()
        script_dir = os.path.dirname(__file__)
        nmodel.parse(os.path.join(script_dir, REL_PATH).format(i + 1))
        network_models.append(nmodel)
    ownship_agent = VcasOwnshipAgent(ACTION_SPACE_DIM, network_models)
    intruder_agent = VcasIntruderAgent(ACTION_SPACE_DIM, network_models)

    env = [
        TwoAgentOwnshipVcasEnv(STATE_SPACE_DIM),
        TwoAgentIntruderVcasEnv(STATE_SPACE_DIM),
        TwoAgentOwnshipIntruderVcasEnv(STATE_SPACE_DIM),
    ][agent_config]

    return [ownship_agent, intruder_agent], env


if __name__ == "__main__":
    main()
