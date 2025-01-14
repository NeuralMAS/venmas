#!/usr/bin/env python
import sys
sys.path.append('../../../')

from src.actors.agents.neural_agent import NeuralAgent
from src.network_parser.network_model import NetworkModel
from src.utils.constants import NeuralOutputSettings
from src.utils.formula import ANextFormula, VarConstConstraint, ConjFormula, StateCoordinate, LT, GT, \
    AtomicConjFormula
from src.verification.bounds.bounds import HyperRectangleBounds
from usecases.frozenlake.frozenlakeenv import FrozenLakeEnv
import numpy as np
import argparse

from src.verification.utils.verification_utils import parallel_single_ctl_verify, mono_single_ctl_verify, seq_single_ctl_verify
import datetime

parser = argparse.ArgumentParser(description="Verify a NANES")
parser.add_argument("-m", "--method", type=int, default=0, help="Method to use for verification: 0. Parallel-poly; 1. Sequential-poly; 2. Mono.")
parser.add_argument("-k", "--steps", type=int, default=3, help="Number of timesteps.")
parser.add_argument("-f", "--formula", type=int, default=1, help="Formula to check.")

ARGS = parser.parse_args()

# Express EF in terms of nested EX.
def get_ag_formula(n, phi):
    formula = ANextFormula(1, phi)
    for i in range(n - 1):
        formula = ANextFormula(1,  ConjFormula(phi, formula))
        # formula = ANextFormula(1, inside)
    return formula
# def get_ag_formula(n, phi):
#     m = 1
#     formula = ANextFormula(m, phi)
#     for i in range(n - 1):
#         m += 1
#         formula = ConjFormula(ANextFormula(m, phi), formula)
#     return formula

def get_ag_until_formula(n, phi, psi):
    formula = ANextFormula(1, psi)
    for i in range(n - 1):
        inside = ConjFormula(phi, formula)
        formula = ANextFormula(1, inside)
    return formula
# def get_ag_until_formula(n, phi, psi):
#     m = 1
#     formula = None
#     for i in range(n - 1):
#         formula = ConjFormula(ANextFormula(m, phi), formula) if formula is not None else ANextFormula(m, phi)
#         m += 1
#     formula = ConjFormula(ANextFormula(m, psi), formula) if formula is not None else ANextFormula(m, psi)
#     return formula


if __name__ == "__main__":

    # Load the agent's neural network representing a robot whose task is to move
    # on a frozen grid world lake and to avoid the two holes in the ice, configured as follows:
    #
    #  F(1) F(2) H(3)
    #  F(4) F(5) F(6)
    #  H(7) F(8) G(9)
    #
    # where in total there are 9 possible states numbered as above, and
    # F stands for a frozen (safe) cell, while H stands for a hole (not safe cell).
    # The initial state is 1, and the goal state is 9.
    #
    # The network gets as input a one-hot encoding of the current environment state--a vector of length 9
    # (with precisely one component being equal one and others being equal zero).
    # The agent produces an action, one of: left (1), down (2), right (3) and up (4)
    # in the form of a one-hot encoding (hence, a vector of length 4).
    agent_model = NetworkModel()
    agent_model.parse("../resources/frozenlake/agent.h5")

    # Load the environment neural networks.
    # The frozen lake environment models the actual physical environment of slippery ice.
    # It does so by, given a direction of desired movement, non-deterministically moving
    # the agent in either the specified direction, or to the right or to the left of it.
    # For example, if the agent was in the following position
    #
    #  1 2 3
    #  4 A 6
    #  7 8 9
    #
    # and desired to move down, then the three possible outcomes would be
    #
    #  1 2 3     1 2 3      1 2 3
    #  A 5 6     4 5 6      4 5 A
    #  7 8 9     7 A 9      7 8 9
    #
    # The networks get as input the one-hot encoding of the current environment state and
    # the one-hot encoding of the agent's action, and return the resulting environment state.
    env_models = []
    for suffix in ["-1", "0", "+1"]:
        env_model = NetworkModel()
        env_model.parse("../resources/frozenlake/env{}.h5".format(suffix))
        env_models.append(env_model)

    # Create a neural agent
    # Since the environment networks accept one-hot encodings of actions,
    # the output setting is set to compute argmax as one hot encoding.
    agent = NeuralAgent(agent_model, output_setting=NeuralOutputSettings.ONE_HOT_ARGMAX)
    # env = NonDeterministicNeuralEnvironment(env_models)
    env = FrozenLakeEnv()

    # The encoding of the initial set of states (a singleton in this case)
    initial_state = np.zeros(agent_model.input_shape)
    initial_state[0] = 1.0
    input_hyper_rectangle = HyperRectangleBounds(initial_state, initial_state)

    # Below are some building blocks of temporal logic specifications we use to verify our robot.

    # All 9 components of an environment state
    coordinates = [StateCoordinate(i) for i in range(9)]

    # These formulas state that the i-th component is strictly less than 0.9.
    # Since in our encoding coordinates can only be 0 or 1,
    # in practice it means that the i-th component is 0
    state_formulas = [VarConstConstraint(coordinates[i], LT, 0.9) for i in range(9)]

    # The following formula describes a state where the second and the sixth components
    # are 0s, that is, a state that is not a hole.
    holes_avoided = AtomicConjFormula(state_formulas[2], state_formulas[6])

    # The following formula describes the goal state.
    goal_reached = VarConstConstraint(coordinates[8], GT, 0.9)

    # Formulas.

    # Verification methods.
    verification_methods = [parallel_single_ctl_verify, seq_single_ctl_verify, mono_single_ctl_verify]
    method_names = ['pool', 'seq', 'mono']

    NUM_STEPS = ARGS.steps

    # Formulas from IJCAI19 paper.
    example_formula_1 = get_ag_formula(NUM_STEPS, holes_avoided)
    example_formula_2 = get_ag_until_formula(NUM_STEPS, holes_avoided, goal_reached)
    formulas = [example_formula_1, example_formula_2]
    # Verification of two families of specifications.
    print(NUM_STEPS, "steps")

    formula = formulas[ARGS.formula - 1]

    print("Formula to verify", formula)
    # Run a method.
    print("Method:", method_names[ARGS.method])

    # Log the info and write to file.
    log_info = verification_methods[ARGS.method](formula, input_hyper_rectangle, agent, env, timeout=3600)
    with open(f"fl-{method_names[ARGS.method]}-{ARGS.formula}.log", "a") as file:
        file.write(
            f"{datetime.datetime.now()}, fl_{NUM_STEPS}, {log_info[0]:9.6f}, {log_info[1]}\n")
    print("")
