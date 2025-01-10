from keras.models import load_model
import collections
import numpy as np
import matplotlib.pyplot as plt

NUM_NETWORKS = 9
BRANCHING_FACTOR = 3

(COC, DNC, DND, DES1500, CL1500, SDES1500, SCL1500, SDES2500, SCL2500) = (0, 1, 2, 3, 4, 5, 6, 7, 8)
(H, H0DOT, TAU) = (0, 1, 2)

input_mean_values = np.array([0.0, 0.0, 20.0])
input_ranges = np.array([16000.0, 5000.0, 40.0])
output_mean = -0.7194709316423972
output_range = 26.24923585890485

adv_2_text = {0: "COC", 1: "DNC", 2: "DND", 3: "DES1500", 4: "CL1500", 5: "SDES1500", 6: "SCL1500", 7: "SDES2500", 8: "SCL2500"}

action_networks = []


def vcas_normalise_input(values):
    return (values - input_mean_values) / input_ranges


def vcas_denormalise_input(values):
    return values * input_ranges + input_mean_values


def vcas_normalise_output(value):
    return (value - output_mean) / output_range


def vcas_denormalise_output(value):
    return value * output_range + output_mean


for i in range(NUM_NETWORKS):
    action_networks.append(load_model("vcas_{}.h5".format(i + 1)))


# prev_adv = CL1500
# example_input = np.array([350.0, -500.0, 35.0])
#
# normalised_input = vcas_normalise_input(example_input)
#
# example_output = action_networks[prev_adv].predict(np.array(normalised_input.reshape(1, 3)))
#
# denormalised_output = vcas_denormalise_output(example_output)
#
# print(adv_2_text[np.argmax(denormalised_output)])


DELTA_T = 1.
G = 32.2


def update_state_vars(position, velocity, acceleration, tau, prev_adv):
    new_position = position - velocity * DELTA_T - 0.5 * acceleration * DELTA_T * DELTA_T
    new_velocity = velocity + acceleration * DELTA_T
    new_tau = tau - DELTA_T
    new_adv = prev_adv
    return new_position, new_velocity, new_tau, new_adv


def plot_example_vcas_policy():
    """
    Reproduce Fig. 9 from (Julian & Kochenderfer, 2019), which shows an example policy plot for
    one of the VerticalCAS networks.
    :return: None
    """
    data = np.zeros((2000, 40))
    for initial_tau in range(0, 40):
        for ipos in range(0, 2000):
            initial_pos = ipos - 1000
            initial_velocity = -900
            initial_acceleration = 3.22

            prev_adv = DES1500

            state = State(initial_pos, initial_velocity, initial_tau, initial_acceleration, prev_adv)

            agent_network = action_networks[state.advisory]

            # 2. Get the next advisory.
            reshaped_input = state.network_input()
            normalised_input = vcas_normalise_input(reshaped_input)
            reshaped_input = normalised_input.reshape(1, 3)
            output = agent_network.predict(np.array(reshaped_input))
            np.set_printoptions(linewidth=np.inf)
            next_adv = vcas_denormalise_output(output)
            next_adv = np.argmax(next_adv)
            data[1999-ipos][initial_tau] = next_adv

    plt.matshow(data, interpolation='nearest', aspect='auto')
    plt.show()


class State(collections.namedtuple('State', 'position velocity tau acceleration advisory')):

    def __str__(self):
        return 'Pos: {} | Vel: {} | Tau: {} | Acc: {} | Adv: {}' \
            .format(self.position, self.velocity, self.tau, self.acceleration, adv_2_text[self.advisory])

    def get_next_state(self, acceleration, next_adv):
        new_position = self.position - self.velocity * DELTA_T - 0.5 * acceleration * DELTA_T * DELTA_T
        new_velocity = self.velocity + acceleration * DELTA_T
        new_tau = self.tau - DELTA_T
        return State(new_position, new_velocity, new_tau, acceleration, next_adv)

    def network_input(self):
        return np.array([self.position, self.velocity, self.tau])

# initial_pos = np.random.uniform(-3000, 3000)
# initial_velocity = np.random.uniform(-2500, 2500)
# initial_tau = np.random.uniform(0, 40)
# initial_acceleration = np.random.uniform(-G / 3, G / 3)

# Plot the example VCAS policy.
# plot_example_vcas_policy()
# exit()


# Specify exact values for the variables for testing purposes.
initial_pos = -130.0
initial_velocity = -26.0
initial_tau = 25
initial_acceleration = 3.22
prev_adv = COC

state = State(initial_pos, initial_velocity, initial_tau, initial_acceleration, prev_adv)
print("Initial state:")
print(state)


def choose_one(acceleration_list):
    return acceleration_list[np.random.choice(BRANCHING_FACTOR)]


while state.tau > 0:
    prev_adv = state.advisory

    # 1. Select network.
    agent_network = action_networks[state.advisory]

    # 2. Get the next advisory.
    reshaped_input = state.network_input()
    normalised_input = vcas_normalise_input(reshaped_input)
    reshaped_input = normalised_input.reshape(1, 3)
    raw_output = agent_network.predict(np.array(reshaped_input))
    # denormalised_output = vcas_denormalise_output(raw_output)

    # next_adv = np.argmax(denormalised_output)
    next_adv = np.argmax(raw_output)

    # 3. Perform business logic in env to get set of next acceleration(s).
    # In practice, we return a list of possible next accelerations/states, but here we draw an
    # acceleration uniformly in the range specified by the given advisory.

    same_advisory_issued = next_adv == prev_adv

    if next_adv == COC:
        next_acceleration = choose_one([-G / 8, 0, G / 8])  # np.random.uniform(-G / 8, G / 8)
    # In the following cases, if next_adv == prev_adv, and the pilot is not compliant, then the
    # pilot will continue choosing an acceleration based on prev_adv (ignoring next_adv),
    # otherwise it does not accelerate (it's compliant), and if prev_adv != next_adv then
    # regardless of whether the pilot is compliant or not, the pilot chooses an advisory based
    # on next_adv and completely ignores prev_adv. This is logically the same as the formula
    # (same_advisory_issued AND compliance) => do_not_accelerate (otherwise accelerate).
    elif next_adv == DNC:
        if same_advisory_issued and state.velocity <= 0:
            next_acceleration = 0  # Maintain constant climbrate.
        else:
            next_acceleration = choose_one([-G / 3, -G * 7 / 24, -G / 4])  # np.random.uniform(-G / 3, -G / 4)
    elif next_adv == DND:
        if same_advisory_issued and state.velocity >= 0:
            next_acceleration = 0
        else:
            next_acceleration = choose_one([G / 4, G * 7 / 24, G / 3])  # np.random.uniform(G / 4, G / 3)
    elif next_adv == DES1500:
        if same_advisory_issued and state.velocity <= -1500:
            next_acceleration = 0
        else:
            next_acceleration = choose_one([-G / 3, -G * 7 / 24, -G / 4])  # np.random.uniform(-G / 3, -G / 4)
    elif next_adv == CL1500:
        if same_advisory_issued and state.velocity >= 1500:
            next_acceleration = 0
        else:
            next_acceleration = choose_one([G / 4, G * 7 / 24, G / 3])  # np.random.uniform(G / 4, G / 3)
    elif next_adv == SDES1500:
        if same_advisory_issued and state.velocity <= -1500:
            next_acceleration = 0
        else:
            next_acceleration = -G / 3
    elif next_adv == SCL1500:
        if same_advisory_issued and state.velocity >= 1500:
            next_acceleration = 0
        else:
            next_acceleration = G / 3
    elif next_adv == SDES2500:
        if same_advisory_issued and state.velocity <= -2500:
            next_acceleration = 0
        else:
            next_acceleration = -G / 3
    elif next_adv == SCL2500:
        if same_advisory_issued and state.velocity >= 2500:
            next_acceleration = 0
        else:
            next_acceleration = G / 3

    # 4. Get the next state.
    state = state.get_next_state(next_acceleration, next_adv)
    print(state)
    print("")

    # 5. State is fed back into agent and loop.
