import math
import random
import matplotlib.pyplot as plt

import networkx as nx
import simpy
import numpy as np
from pprint import pprint as pp
from numpy.lib.stride_tricks import sliding_window_view


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
                self.period = self.next_period
                self.next_period = DEFAULT_PERIOD_LENGTH
                self.highest_msg_this_epoch = (self.id, self.epoch)

            self.log_phase()
            self.log_epoch()

            # update local timer (can be made stochastic)
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
            reception_prob = reception_probability(distance)
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
            node_phase_x[self.id].append(self.env.now)
            node_phase_y[self.id].append(0)
            node_phase_percentage_x[self.id].append(self.env.now)
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
# test_params = {
#     'TRIAL': True,
#     'LOGGING': False,
#     'RANDOM_SEED': 11,
#     'OVERALL_MULT': 1000,
#     'RECEPTION_LOOP_TICKS': OVERALL_MULT / 10,
#     'DEFAULT_PERIOD_LENGTH': 100 * OVERALL_MULT,
#     'SIM_TIME': 2000 * OVERALL_MULT,
#
#     'MS_PROB': 1,  # 0.8  # 1
#
#     'M_TO_PX': 90,  ########## ACCURATE: 100    DENSE: 150      BUGGY: 120
#     'DISTANCE_EXPONENT': 15,  # ACCURATE: 10     DENSE: 8        BUGGY: 3
#
#     'CLOCK_DRIFT_RATE_OFFSET_RANGE': 100,
#     'CLOCK_DRIFT_VARIABILITY': 0.05,
#
#     'MIN_INITIAL_TIME_OFFSET': 0,
#     'MAX_INITIAL_TIME_OFFSET': 250,
#
#     'NUM_NODES': 41,
#     'SYNC_EPSILON': 2  # (percent of period length),
# }
########## CONSTANTS ##########
TRIAL = True
LOGGING = False
RANDOM_SEED = 11
OVERALL_MULT = 1000
RECEPTION_LOOP_TICKS = OVERALL_MULT / 10
DEFAULT_PERIOD_LENGTH = 100 * OVERALL_MULT
SIM_TIME = 2000 * OVERALL_MULT

MS_PROB = 1  # 0.8  # 1

M_TO_PX = 90  ########## ACCURATE: 100    DENSE: 150      BUGGY: 120
DISTANCE_EXPONENT = 15  # ACCURATE: 10     DENSE: 8        BUGGY: 3

CLOCK_DRIFT_RATE_OFFSET_RANGE = 100
CLOCK_DRIFT_VARIABILITY = 0.05

MIN_INITIAL_TIME_OFFSET = 0
MAX_INITIAL_TIME_OFFSET = 250

NUM_NODES = 5
SYNC_EPSILON = 2  # (percent of period length)
SYNC_NUM_TICKS = 1.0  # fraction of next 100 ticks that must be within epsilon
########## CONSTANTS ##########


# edge_list = [(kvs[0], v) for kvs in topo.items() for v in kvs[1]]
# G = nx.from_edgelist(edge_list)

# Generate graph plot and positions
# topo_gen = nx.random_internet_as_graph
topo_gen = nx.complete_graph

G = topo_gen(NUM_NODES)  # topo_gen(15, 1)
pos = nx.nx_agraph.graphviz_layout(G)

# node configuration
init_state = [
    {
        'id': i,
        'initial_time_offset': random.randint(MIN_INITIAL_TIME_OFFSET, MAX_INITIAL_TIME_OFFSET),
        'clock_drift_rate': random.randint(-CLOCK_DRIFT_RATE_OFFSET_RANGE, CLOCK_DRIFT_RATE_OFFSET_RANGE),
        'clock_drift_scale': CLOCK_DRIFT_VARIABILITY
    }
    for i in range(NUM_NODES)
]

reception_probabilities = {i: {} for i in range(NUM_NODES)}
for x, i in enumerate(pos):
    for y, j in enumerate(pos):
        if i != j:
            reception_prob = reception_probability(distance_euc(pos[i], pos[j]))
            reception_probabilities[x][y] = round(reception_prob, 3)

# if LOGGING:
print('reception_probabilities:')
pp(reception_probabilities)

nx.draw(G, with_labels=True, pos=pos)

# Set up data structures for logging
node_phase_x = [[] for _ in range(NUM_NODES)]
node_phase_y = [[] for _ in range(NUM_NODES)]

node_phase_percentage_x = [[] for _ in range(NUM_NODES)]
node_phase_percentage_y = [[] for _ in range(NUM_NODES)]

node_epochs_x = [[] for _ in range(NUM_NODES)]
node_epochs_y = [[] for _ in range(NUM_NODES)]
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
for i in range(NUM_NODES):
    ax[0].plot(node_phase_x[i], node_phase_y[i], label='node ' + str(i), linewidth=2,
               linestyle='dashdot')
ax[0].set_title('Node phase')
ax[0].set(
    ylabel='Time since last fire')  # xlabel='Time (ms)', ylabel='Phase (relative to default period length = 100s)')
ax[0].legend(loc="upper right")

# # Fires and suppresses
ax[1].plot(reception_x, reception_y, '*', color='blue', label='message reception', markersize=7, alpha=0.3)
ax[1].plot(suppress_x, suppress_y, 'x', color='grey', label='node suppress', markersize=5)
ax[1].plot(fire_x, fire_y, 'o', color='red', label='node fire', markersize=5)

ax[1].set_xticks(np.arange(0, SIM_TIME + 1, DEFAULT_PERIOD_LENGTH))
ax[1].grid()
ax[1].set_title('Fires and suppresses')
ax[1].set(ylabel='Node ID')
ax[1].legend(loc="upper right")

# Node epoch
for i in range(NUM_NODES):
    ax[2].plot(node_epochs_x[i], node_epochs_y[i], label='node ' + str(i), linestyle='dashdot',
               linewidth=2)
ax[2].set_title('Node epoch')
ax[2].set(ylabel='Node epoch')
ax[2].legend(loc="upper right")

fig.suptitle('Modified PCO Node Simulation for a ' + topo_gen.__name__ + ' topology with ' + str(NUM_NODES) + ' nodes')

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
max_differences = np.max(differences, axis=0)

avg_phase_diff_after_synchronization = np.round(np.mean(mean_differences[-100:]), 2)
time_until_synchronization = SIM_TIME

# Find time until synchronization
"""We consider the network synchronized when the maximum pair-wise phase difference between all nodes at a given tick 
is less than 2% of the period length, and remains so for 95% of the subsequent 100 ticks"""
v = sliding_window_view(max_differences, 1000)
for i, window in enumerate(v):
    if (window < SYNC_EPSILON).sum() >= SYNC_NUM_TICKS * len(window):
        time_until_synchronization = i * RECEPTION_LOOP_TICKS
        break

ax[3].plot(x, mean_differences, label='mean phase difference', linewidth=2)
ax[3].plot(x, max_differences, label='max phase difference', linewidth=2)
ax[3].set_title('Phase difference')
ax[3].set(xlabel='Time (ms)', ylabel='Pair-wise phase difference (%)')
ax[3].axvline(x=time_until_synchronization, color='blue', label='Synchronization time', linewidth=2, linestyle='--',
              alpha=0.5, marker='o')
ax[3].legend(loc="upper right")

time_until_synchronization_human_readable = np.round(time_until_synchronization / 1000, 2)

# Print out metrics for the network
print('Number of broadcasts:', num_broadcasts)
print("Synchronized avg. phase diff: ", avg_phase_diff_after_synchronization)
print("Time until synchronization: ", time_until_synchronization_human_readable)

metrics = [
    "Number of broadcasts: " + str(num_broadcasts),
    "Time until synchronization: " + str(time_until_synchronization_human_readable),
    "Synchronized avg. phase diff: " + str(avg_phase_diff_after_synchronization) + '%',
    "Topo: " + str(topo_gen.__name__),
    "Message Suppression prob.: " + str(MS_PROB * 100) + '%',
    "Non-adj. node comm. prob. params.: C=" + str(M_TO_PX) + ' E=' + str(DISTANCE_EXPONENT),
]

for i, metric_text in enumerate(metrics):
    plt.text(.50, .96 - .1 * i, metric_text, ha='left', va='top', transform=ax[3].transAxes)

plt.show()
