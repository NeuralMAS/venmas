from keras.models import load_model
import collections
import numpy as np

NUM_NETWORKS = 1
BRANCHING_FACTOR = 3
action_to_array = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

def st_array_to_txt(a):
    txt = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    a1 = a == np.array(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]))
    a2 = a == np.array(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]))
    a3 = a == np.array(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]))
    a4 = a == np.array(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]))
    a5 = a == np.array(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]))
    a6 = a == np.array(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]))
    a7 = a == np.array(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]))
    a8 = a == np.array(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]))
    a9 = a == np.array(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]))
    for i, n in enumerate([a1, a2, a3, a4, a5, a6, a7, a8, a9]):
        if n.all():
            return txt[i]
def array_to_txt(a):
    txt = ["left", "down", "right", "up"]
    a1 = a == np.array(np.array([1, 0, 0, 0]))
    a2 = a == np.array(np.array([0, 1, 0, 0]))
    a3 = a == np.array(np.array([0, 0, 1, 0]))
    a4 = a == np.array(np.array([0, 0, 0, 1]))
    for i, n in enumerate([a1, a2, a3, a4]):
        if n.all():
            return txt[i]

action_network = load_model("agent.h5")


class State(collections.namedtuple('State', 'state action')):

    def __str__(self):
        return 'state: {} | action: {}' \
            .format(st_array_to_txt(self.state), array_to_txt(self.action))

    def network_input(self):
        return np.array([self.state])

# initial_pos = np.random.uniform(-3000, 3000)
# initial_velocity = np.random.uniform(-2500, 2500)
# initial_tau = np.random.uniform(0, 40)
# initial_acceleration = np.random.uniform(-G / 3, G / 3)

# Plot the example VCAS policy.
# plot_example_vcas_policy()
# exit()


# Specify exact values for the variables for testing purposes.


prev_action = [1, 0, 0, 0]
initial_state = np.zeros(9)
initial_state[0] = 1.0
initial_state = initial_state.reshape(1, 9)

state = State(initial_state, prev_action)
print("Initial state:")
print(st_array_to_txt(state.state))


def choose_one(acceleration_list):
    return acceleration_list[np.random.choice(BRANCHING_FACTOR)]


env_models = []
for suffix in ["-1", "0", "+1"]:
    env_models.append(load_model("env{}.h5".format(suffix)))
tau = 10
while tau > 0:
    prev_action = state.action

    # 1. Select network.
    agent_network = action_network

    # 2. Get the next advisory.
    reshaped_input = state.network_input()
    reshaped_input = reshaped_input.reshape(1, 9)
    raw_output = agent_network.predict(np.array(reshaped_input))
    raw_output = raw_output.reshape(1, 4)
    # denormalised_output = vcas_denormalise_output(raw_output)

    # next_adv = np.argmax(denormalised_output)
    next_action = np.array(action_to_array[np.argmax(raw_output)]).reshape(1, 4)

    # 3. Perform business logic in env to get set of next acceleration(s).
    # In practice, we return a list of possible next accelerations/states, but here we draw an
    # acceleration uniformly in the range specified by the given advisory.

    transition_state_input = np.concatenate((state.state, next_action), axis=1)
    reshaped_transition_input = transition_state_input.reshape(1, 13)
    transition = choose_one([env_models[0].predict(np.array(reshaped_transition_input)),
                             env_models[1].predict(np.array(reshaped_transition_input)),
                             env_models[2].predict(np.array(reshaped_transition_input))])

    tau = tau - 1
    # 4. Get the next state.
    state = State(transition, next_action)
    print(state)
    print("")

    # 5. State is fed back into agent and loop.

