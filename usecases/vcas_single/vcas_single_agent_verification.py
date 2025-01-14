#!/usr/bin/env python

import sys
sys.path.append('../../../')

import argparse
import datetime
import os

from usecases.vcas.vcasagent import VcasAgent
from usecases.vcas.vcasenv import VcasEnv, VcasConstants
from src.network_parser.network_model import NetworkModel
from src.utils.formula import *
from src.verification.bounds.bounds import HyperRectangleBounds

from src.verification.utils.verification_utils import mono_single_ctl_verify, parallel_single_ctl_verify, seq_single_ctl_verify

parser = argparse.ArgumentParser(description="Verify a NANES")
parser.add_argument("-m", "--method", type=int, default=2, help="Method to use for verification: 0. Parallel-poly; 1. Sequential-poly; 2. Mono.")
parser.add_argument("-k", "--steps", type=int, default=4, help="Number of timesteps.")
parser.add_argument("-c", "--climbrate", type=float, default=-19.5, help="Climbrate.")

ARGS = parser.parse_args()


def initialise_and_get_agent_and_env():
    """
    Initialise agent and environment.
    :return: Initialised VcasAgent and VcasEnv objects.
    """

    # Where the agent networks are.
    REL_PATH = "vcas_single/models/vcas_{}.h5"

    network_models = []
    for i in range(VcasConstants.N_ADVISORIES):
        nmodel = NetworkModel()
        script_dir = os.path.dirname(__file__)
        nmodel.parse(os.path.join(script_dir, REL_PATH).format(i + 1))
        network_models.append(nmodel)
    agent = VcasAgent(network_models)
    env = VcasEnv()

    return agent, env


def main():
    # Constraint specific variables of the initial state to one value by setting the upper
    # bounds equal to the lower bounds.
    initial_advisory = VcasConstants.COC
    initial_acceleration = VcasConstants.G / 10
    initial_tau = 25
    initial_altitude = -131
    noise = 2

    agent, env = initialise_and_get_agent_and_env()

    verification_methods = [parallel_single_ctl_verify, seq_single_ctl_verify, mono_single_ctl_verify]
    method_names = ['pool', 'seq', 'mono']

    initial_climbrate = ARGS.climbrate
    steps = ARGS.steps

    print("###########", steps, "time steps;", initial_climbrate, "initial climbrate ##########")
    safe = AtomicDisjFormula(VarConstConstraint(StateCoordinate(VcasConstants.INTRUDER_ALTITUDE), GT, 100),
                             VarConstConstraint(StateCoordinate(VcasConstants.INTRUDER_ALTITUDE), LT, -100))
    safety_formula = ANextFormula(steps, safe)

    unzipped = zip(*[(initial_altitude - noise, initial_altitude + noise),
                     (initial_climbrate,        initial_climbrate),
                     (initial_tau,              initial_tau),
                     (initial_acceleration,     initial_acceleration),
                     (initial_advisory,         initial_advisory)])

    input_hyper_rectangle = HyperRectangleBounds(*unzipped)
    print("")
    log_info = verification_methods[ARGS.method](safety_formula, input_hyper_rectangle, agent, env, timeout=3600)
    with open(f"vcas_single-{method_names[ARGS.method]}.log", "a") as file:
        file.write(
            f"{datetime.datetime.now()}, VCAS_{initial_climbrate}, {steps}, {log_info[0]:9.6f}, {log_info[1]}\n")

    print("")


if __name__ == "__main__":
    main()
