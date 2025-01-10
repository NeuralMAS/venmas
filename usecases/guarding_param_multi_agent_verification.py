#!/usr/bin/env python

import argparse
import sys
sys.path.append('../')

from resources.guarding.guardagent import GuardingAgent, GuardingConstants
from resources.guarding.guardzoagent import GuardingZeroOneAgent
from resources.guarding.guardenv import GuardingEnv
from src.network_parser.network_model import NetworkModel
from src.utils.formula import *
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.utils.verification_utils import mono_multi_ctl_verify


def main():
    parser = argparse.ArgumentParser(description="Verify a MANS")
    parser.add_argument("-f", "--formula", type=int, default=0, help="Formula to verify: 0. EX^k alive; 1. AX^k alive;")
    parser.add_argument("-k", "--step", default=4, type=int, help="The number of time steps to verify for.")
    parser.add_argument("-a", "--agents_number", default=2, type=int, help="Number of template agents.")
    parser.add_argument("-n", "--threshold", default=3, type=int, help="Number of template agents.")
    parser.add_argument("-m", "--method", type=int, default=0, help="Method to run: 0. Monolithic.")
    parser.add_argument("-hp", "--initial_health", default=3, type=int, help="Initial health points of a template agent.")
    parser.add_argument("-per", "--initial_percept", default=2, type=int, help="Initial percept of a template agent (one of 0-expired, 1-rest, or 2-volunteer-to-guard).")
    parser.add_argument("-w", "--workers", default=2, type=int, help="Number of workers.")
    parser.add_argument("-to", "--timeout", default=3600, type=int, help="Timeout in minutes.")

    ARGS = parser.parse_args()

    ############################################################################
    print(f"========== Parameters of the game:==========\n"
          f"\t Guarding reward: {GuardingConstants.GUARDING_REWARD}\n"
          f"\t  Resting reward: {GuardingConstants.RESTING_REWARD}\n"
          f"\tUnguarded reward: {GuardingConstants.UNGUARDED_REWARD}\n"
          f"\t  Maximum health: {GuardingConstants.MAX_HEALTH_POINTS}\n")

    print(f"Template agent number: {ARGS.agents_number}\n")


    ############################################################################
    # Verify for the given number of steps
    if ARGS.formula == 0:
        steps = [ARGS.threshold]#range(ARGS.agents_number, ARGS.threshold + 1)
        for agent_count in steps:
            print(agent_count, "agents in NIS")

            agents, env = initialise_and_get_agent_and_env(agent_count, ARGS.formula)

            input_hyper_rectangle = get_input_bounds(agent_count, ARGS.formula, ARGS.initial_health, ARGS.initial_percept)
            print(input_hyper_rectangle, "\n")

            formula = get_formula(ARGS.formula, ARGS.step, ARGS.agents_number)

            print("Formula to verify", formula)
            # Run a method.
            verification_methods = [mono_multi_ctl_verify]
            verification_methods[ARGS.method](formula, input_hyper_rectangle, agents, env, ARGS.timeout)
            print("\n")

    else:
        steps = [ARGS.step]
        for temp_depth in steps:
            print(temp_depth, "time steps")

            agents, env = initialise_and_get_agent_and_env(ARGS.agents_number, ARGS.formula)

            input_hyper_rectangle = get_input_bounds(ARGS.agents_number, ARGS.formula, ARGS.initial_health,
                                                     ARGS.initial_percept)
            print(input_hyper_rectangle, "\n")

            formula = get_formula(ARGS.formula, temp_depth, ARGS.agents_number)

            print("Formula to verify", formula)
            # Run a method.
            verification_methods = [mono_multi_ctl_verify]
            verification_methods[ARGS.method](formula, input_hyper_rectangle, agents, env, ARGS.timeout)
            print("\n")


def get_formula(formula, num_steps, agents_number):
    if formula == 0:
        clauses = [
            VarConstConstraint(
                StateCoordinate(GuardingConstants.HEALTH_IDX + i * GuardingConstants.AGENT_STATE_DIMENSIONS),
                GE, GuardingConstants.EXPIRED_HEALTH_POINTS + 1)
            for i in range(agents_number)
        ]
        return ENextFormula(num_steps, NAryConjFormula(clauses))

    elif formula == 1:
        clauses = [
            VarConstConstraint(
                StateCoordinate(GuardingConstants.HEALTH_IDX + i * GuardingConstants.AGENT_STATE_DIMENSIONS),
                GT, GuardingConstants.EXPIRED_HEALTH_POINTS)
            for i in range(agents_number)
        ]
        return ANextFormula(num_steps, NAryConjFormula(clauses))

    return None


def get_input_bounds(agents_number, formula, initial_health, initial_percept):
    ###########################################################################
    # Compute the initial state, will be used for the lower and upper bounds. #
    ###########################################################################
    initial_state = []

    # The template agent components
    for agent in range(agents_number):
        initial_state.extend([initial_health, initial_percept])

    if formula == 1:
        # Zero-one initial values
        # no expired
        initial_state.append(0)
        # only one value of health
        initial_state.extend([0] * (initial_health-1) +
                             [1] +
                             [0] * (GuardingConstants.MAX_HEALTH_POINTS - initial_health))
        # zero-one percepts
        initial_state.extend([0, 0, 0])

    # The environment part
    initial_state.append(0)

    return HyperRectangleBounds(initial_state, initial_state)


def initialise_and_get_agent_and_env(agents_number, formula):
    """
    Initialise agent and environment.
    :param formula whether the formula is existential 0 or universal 1
    :return: List of initialised GuardingAgent objects and a GuardingEnv object.
    """

    # Where the agent networks are.
    REL_PATH = "../resources/guarding/agent.h5"

    network_model = NetworkModel()
    network_model.parse(REL_PATH)

    agents = [GuardingAgent(network_model) for _ in range(agents_number)]

    if formula == 1:
        agents += [GuardingZeroOneAgent(network_model)]

    env = GuardingEnv()

    return agents, env


if __name__ == "__main__":
    main()
