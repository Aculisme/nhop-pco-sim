import math
import random
import matplotlib.pyplot as plt

import networkx as nx
import simpy
import statistics
import numpy as np
from pprint import pprint as pp


class PCONode7:
    """Adding stochasticity to links and timers"""

    def __init__(self, env, init_state):  # , state):
        """Initialise the node with a given *env* and *state*."""

        self.id = init_state['id']
        self.state = init_state
        self.env = env

        # set externally
        self.neighbors = None
        self.pos = None

        # externally accessible
        self.buffer = []

        # used by main loop
        self.epoch = 0
        self.highest_msg_this_epoch = (self.id, self.epoch)
        self.period = DEFAULT_PERIOD_LENGTH
        self.next_period = DEFAULT_PERIOD_LENGTH
        self.timer = 0
        # self.messages_seen_this_epoch = 0

        self.MS_PROB = MS_PROB
        self.clock_drift_rate = init_state['clock_drift_rate']
        self.clock_drift_scale = init_state['clock_drift_scale']

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting
        yield self.env.timeout(self.state['initial_time_offset'] * OVERALL_MULT)

        while True:

            while self.buffer:

                new_msg = self.buffer.pop()
                # self.log('received message from node', new_msg[0], 'with epoch', new_msg[1])

                # If the new msg epoch is greater than the largest epoch message seen this period previously, update
                #   and sync to it
                # This will happen when all nodes are synced, since epoch is incremented before sending
                if new_msg[1] > self.highest_msg_this_epoch[1]:
                    self.highest_msg_this_epoch = new_msg

                    # synchronize to it
                    self.next_period = DEFAULT_PERIOD_LENGTH - (self.period - self.timer)  # - 1*OVERALL_MULT
                    # next period should be the time since the todo
                    self.log('synchronized to node', new_msg[0], 'which has epoch', new_msg[1],
                             'setting next period to', self.next_period)

                elif new_msg[1] == self.highest_msg_this_epoch[1]:
                    # new msg has same epoch but message arrived later, ignore.
                    # self.log('ignoring message from node', new_msg[0], 'with epoch', new_msg[1])
                    pass

                if new_msg[1] >= self.epoch + 2:
                    # we're super out of sync, match their epoch number
                    self.log("super out of sync, matching epoch number")
                    self.epoch = new_msg[1] - 1

            # timer expired
            if self.timer >= self.period:

                # increment epoch now that our timer has expired
                self.epoch += 1

                # if no message seen this epoch, broadcast
                if self.highest_msg_this_epoch[0] == self.id:

                    self.log('fired: broadcast')
                    self._tx((self.id, self.epoch))

                # todo: chance of ignoring suppression, depending on randomness OR connectivity?
                elif random.random() > self.MS_PROB:
                    self.log('fired: broadcast (MS override)')
                    self._tx((self.id, self.epoch))

                else:
                    # suppress message
                    self.log_suppress()
                    self.log('fired: suppressed')
                    pass

                self.timer = 0
                # self.out_of_sync = False
                self.period = self.next_period
                self.next_period = DEFAULT_PERIOD_LENGTH
                self.highest_msg_this_epoch = (self.id, self.epoch)

            self.log_phase()
            self.log_epoch()

            # update local timer (can be made stochastic)
            # tick_len = int(rng.normal(1 + self.clock_drift_rate * 1e-6, 0.001) * OVERALL_MULT)  # self.clock_drift_scale
            tick_len = int(rng.normal(1 * RECEPTION_LOOP_TICKS + self.clock_drift_rate * 3e-6 * RECEPTION_LOOP_TICKS,
                                      self.clock_drift_scale * RECEPTION_LOOP_TICKS))  # self.clock_drift_scale
            # print(tick_len)
            # self.timer += tick_len
            self.timer += RECEPTION_LOOP_TICKS  # * OVERALL_MULT // 2

            self.log_phase_helper()

            # yield self.env.timeout(1)
            yield self.env.timeout(tick_len)

    def _tx(self, message):
        """Broadcast a *message* to all receivers."""
        self.log_fire()
        if not self.neighbors:
            raise RuntimeError('There are no neighbors to send to.')

        for neighbor in self.neighbors:
            distance = distance_euc(self.pos, neighbor.pos)
            # reception_prob = 1 / (1 + distance ** 2)
            # reception_prob = 1 * math.e ** (-(distance ** 2))
            reception_prob = reception_probability(distance)
            # print('from', self.id, 'to', neighbor.id, ':', reception_prob)

            # message reception probability proportional to inverse distance squared
            # if random.random() < NEIGHBOR_RECEPTION_PROB:
            if random.random() < reception_prob:
                neighbor.buffer.append(message)

                self.log_reception(neighbor)

    def log(self, *message):
        """Log a message with the current time and node id."""
        if LOGGING:
            print('node', self.id, '|', 'time', self.env.now / OVERALL_MULT, '| epoch', self.epoch, '|', *message)

    def log_phase(self):
        node_phase_x[self.id].append(self.env.now)
        node_phase_y[self.id].append(self.timer)
        node_phase_percentage_x[self.id].append(self.env.now)
        node_phase_percentage_y[self.id].append((self.timer / max(self.period, 1)) * 100)

    def log_phase_helper(self):
        """needed to instantly set next env tick phase to 0, otherwise waits until next large tick to set to zero,
        messing up the interpolation when graphing"""
        if self.timer >= self.period:
            node_phase_x[self.id].append(self.env.now + 1)
            node_phase_y[self.id].append(0)
            node_phase_percentage_x[self.id].append(self.env.now + 1)
            node_phase_percentage_y[self.id].append(0)

    def log_fire(self):
        fire_x.append(self.env.now)
        fire_y.append(self.id)

    def log_suppress(self):
        suppress_x.append(self.env.now)
        suppress_y.append(self.id)

    def log_epoch(self):
        node_epochs_x[self.id].append(self.env.now)
        node_epochs_y[self.id].append(self.epoch)

    def log_reception(self, neighbor):
        reception_x.append(self.env.now)
        reception_y.append(neighbor.id)


def distance_euc(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 / M_TO_PX


def reception_probability(distance):
    return 1 / (1 + distance ** DISTANCE_EXPONENT)


env = simpy.Environment()

# Topology Configuration

TOPO_FULLY_CONNECTED = {
    0: [1, 2, 3, 4, 5],
    1: [0, 2, 3, 4, 5],
    2: [0, 1, 3, 4, 5],
    3: [0, 1, 2, 4, 5],
    4: [0, 1, 2, 3, 5],
    5: [0, 1, 2, 3, 4],
}

TOPO_BRIDGE = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2, 4, 5],
    4: [3, 5],
    5: [3, 4],
}

TOPO_MINI_FC = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1],
}

TOPO_FOUR_BRIDGES = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 10],

    3: [4, 5, 1],
    4: [3, 5],
    5: [3, 4, 6],

    6: [7, 8, 5],
    7: [6, 8],
    8: [6, 7, 9],

    9: [10, 11, 8],
    10: [9, 11, 2],
    11: [9, 10],
}

# TOPO_FULLY_CONNECTED = {}
# add another 50 nodes to the fully connected topo and the init state variable
# for i in range(50):
#     TOPO_FULLY_CONNECTED[i] = [j for j in range(50) if j != i]
#     init_state.append(
#         {'id': i, 'initial_time_offset': random.randint(0, 500), 'clock_drift_rate': random.randint(-10, 10),
#          'clock_drift_scale': 0.1, 'linestyle': 'dashdot', 'hatch': 'x'})


########## CONSTANTS ##########
# topo = TOPO_FULLY_CONNECTED
# topo = TOPO_MINI_FC
topo = TOPO_BRIDGE
# topo = TOPO_FOUR_BRIDGES

LOGGING = False
RANDOM_SEED = 10
OVERALL_MULT = 1000
RECEPTION_LOOP_TICKS = OVERALL_MULT / 10
DEFAULT_PERIOD_LENGTH = 100 * OVERALL_MULT
SIM_TIME = 2000 * OVERALL_MULT

MS_PROB = 0.8  # 1

M_TO_PX = 100  ########## ACCURATE: 100    DENSE: 150      BUGGY: 120
DISTANCE_EXPONENT = 10  # ACCURATE: 10     DENSE: 8        BUGGY: 3

CLOCK_DRIFT_RATE_OFFSET_RANGE = 100
CLOCK_DRIFT_VARIABILITY = 0.05
########## CONSTANTS ##########

# node configuration
init_state = [
    {'id': i, 'initial_time_offset': random.randint(0, 500),
     'clock_drift_rate': random.randint(-CLOCK_DRIFT_RATE_OFFSET_RANGE, CLOCK_DRIFT_RATE_OFFSET_RANGE),
     'clock_drift_scale': CLOCK_DRIFT_VARIABILITY,
     'linestyle': 'dashdot'}
    for i in topo
]

# Generate graph plot and positions
edge_list = [(kvs[0], v) for kvs in topo.items() for v in kvs[1]]
G = nx.from_edgelist(edge_list)
pos = nx.nx_agraph.graphviz_layout(G)

reception_probabilities = {i['id']: {} for i in init_state}
for x, i in enumerate(pos):
    for y, j in enumerate(pos):
        if i != j:
            reception_prob = reception_probability(distance_euc(pos[i], pos[j]))
            reception_probabilities[x][y] = round(reception_prob, 3)

print('reception_probabilities:')
pp(reception_probabilities)

nx.draw(G, with_labels=True, pos=pos)

# Set up data structures for logging
node_phase_x = [[] for _ in init_state]
node_phase_y = [[] for _ in init_state]

node_phase_percentage_x = [[] for _ in init_state]
node_phase_percentage_y = [[] for _ in init_state]

node_epochs_x = [[] for _ in init_state]
node_epochs_y = [[] for _ in init_state]
fire_x = []
fire_y = []
suppress_y = []
suppress_x = []
reception_x = []
reception_y = []

random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

nodes = [PCONode7(env, state) for state in init_state]

for i, node in enumerate(nodes):
    node.neighbors = [nodes[n] for n in range(len(nodes)) if n != i]  # [nodes[n] for n in topo[i]]
    node.pos = pos[i]

env.run(until=SIM_TIME)

# Generate plots

# Generate synchronization plots
fig, ax = plt.subplots(4, sharex=True)

# Node phase
for i in init_state:
    ax[0].plot(node_phase_x[i['id']], node_phase_y[i['id']], label='node ' + str(i['id']), linewidth=2,
               linestyle=i['linestyle'])  # , alpha=0.4) #
ax[0].set_title('Node phase')
ax[0].set(
    ylabel='Time since last fire')  # xlabel='Time (ms)', ylabel='Phase (relative to default period length = 100s)')
ax[0].legend(loc="upper right")

# # Fires and suppresses
ax[1].plot(reception_x, reception_y, '*', color='blue', label='message reception', markersize=7, alpha=0.3)
ax[1].plot(suppress_x, suppress_y, 'x', color='grey', label='node suppress', markersize=5)
ax[1].plot(fire_x, fire_y, 'o', color='red', label='node fire', markersize=5)

# ax[1].grid(axis='y')
ax[1].set_xticks(np.arange(0, SIM_TIME + 1, DEFAULT_PERIOD_LENGTH))
ax[1].grid()
ax[1].set_title('Fires and suppresses')
ax[1].set(ylabel='Node ID')
ax[1].legend(loc="upper right")

# Node epoch
for i in init_state:
    ax[2].plot(node_epochs_x[i['id']], node_epochs_y[i['id']], label='node ' + str(i['id']), linestyle=i['linestyle'],
               linewidth=2)
ax[2].set_title('Node epoch')
ax[2].set(ylabel='Node epoch')
ax[2].legend(loc="upper right")

fig.suptitle('Initial time offsets: ' + str([i['initial_time_offset'] for i in init_state]))

num_broadcasts = len(fire_x)


def calculate_phase_difference(wave1, wave2, max_phase_percentage=100):
    diff = np.abs(wave1 - wave2)
    diff_opposite = abs(max_phase_percentage - diff)
    return np.minimum(diff, diff_opposite)


x = np.linspace(0, SIM_TIME, int(SIM_TIME / RECEPTION_LOOP_TICKS))

interpolated = np.array([np.interp(x, node_phase_percentage_x[i], node_phase_percentage_y[i]) for i in
                         range(len(node_phase_percentage_y))])

differences = []
# todo: inefficient to do pairwise calcs, change if we're using huge topologies
# O(n(n-1)/2) complexity :(
for i in range(len(node_phase_percentage_y)):
    for j in range(i + 1, len(node_phase_percentage_y)):
        differences.append(calculate_phase_difference(interpolated[i], interpolated[j]))

differences = np.array(differences)
mean_differences = np.mean(differences, axis=0)

ax[3].plot(x, mean_differences, label='mean_differences', linewidth=2)
ax[3].set_title('Average Phase difference')
ax[3].set(xlabel='Time (ms)', ylabel='Pair-wise avg. phase difference (%)')
ax[3].legend(loc="upper right")

# Print out metrics for the network
print('Number of broadcasts:', num_broadcasts)
# print('Average synch range (ticks):', avg_range)
# print('Average synch st. dev. (ticks):', avg_stdev)
print("Avg. phase difference after synchronization", np.mean(mean_differences[-100:]))

# plt.text(-1, -1, "Number of broadcasts: " + str(num_broadcasts))
plt.text(.80, .90, "Number of broadcasts: " + str(num_broadcasts), ha='left', va='top', transform=ax[3].transAxes)
# plt.text(.80, .80, "Avg. synch range (ticks): " + str(avg_range), ha='left', va='top', transform=ax[3].transAxes)
# todo: add topo

# for x in range(7350, 7850):
#     print(interpolated[0][x], interpolated[1][x], interpolated[2][x], interpolated[3][x], interpolated[4][x], interpolated[5][x], differences[4][x])


plt.show()
