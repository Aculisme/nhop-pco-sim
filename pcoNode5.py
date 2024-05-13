import random
import matplotlib.pyplot as plt

import networkx as nx
import simpy
import statistics
import numpy as np


class PCONode5:
    """Adding stochasticity to links and timers"""

    def __init__(self, env, init_state):  # , state):
        """Initialise the node with a given *env* and *state*."""
        # set externally
        self.epoch = 0
        self.neighbors = None

        # externally accessible
        # self.buffer = simpy.Store(env)
        self.buffer = []

        self.id = init_state['id']
        self.state = init_state
        self.env = env
        # self.pos = init_state.pos

        self.period = DEFAULT_PERIOD_LENGTH
        self.next_period = DEFAULT_PERIOD_LENGTH
        self.timer = 0
        self.messages_seen_this_epoch = 0

        # self.SEND_PROB = 0.80
        self.SEND_PROB = 1
        self.MS_PROB = MS_PROB
        self.clock_drift_rate = init_state['clock_drift_rate']
        self.clock_drift_scale = init_state['clock_drift_scale']

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting
        yield self.env.timeout(self.state['initial_time_offset'])

        while True:

            # look at received messages, and decide whether to broadcast
            if self.buffer:  # max(self.buffer) > self.timer:

                self.log('buffer:', self.buffer,
                         'timer:', self.timer,
                         'period:', self.period,
                         'next period:', self.next_period,
                         'epoch:', self.epoch,
                         'messages seen this epoch:', self.messages_seen_this_epoch)

                # todo: investigate these two diff cases
                if self.buffer[0][1] > self.epoch:
                    self.next_period = DEFAULT_PERIOD_LENGTH - (self.period - self.timer) - 1
                    self.log('case 0: setting next period to:', self.next_period, 'from node', self.buffer[0][0],
                             "'s message")
                    self.messages_seen_this_epoch += 1
                    # self.epoch = self.buffer[0][1]

                elif self.timer > DEFAULT_PERIOD_LENGTH / 2 and self.buffer[0][1] >= self.epoch:
                    # shorten next period by the time remaining in the current period (?)
                    self.next_period = DEFAULT_PERIOD_LENGTH - (self.period - self.timer) - 1
                    self.log('case 1: setting next period to:', self.next_period, 'from node', self.buffer[0][0],
                             "'s message")
                    self.messages_seen_this_epoch += 1
                    # self.epoch = self.buffer[0][1]

                self.buffer.clear()

            # timer expired
            if self.timer >= self.period:

                self.epoch += 1  # Different from pcoNode4.py

                # broadcast
                if self.messages_seen_this_epoch < 1 or random.random() > self.MS_PROB:  # FiGo EXTENSION
                    self.log('broadcast')
                    self._tx((self.id, self.epoch))

                    self.log_fire()
                    self.log('fired')

                # suppress message
                else:
                    self.log_suppress()

                # reset timer & prepare for next period
                self.timer = 0
                self.messages_seen_this_epoch = 0
                self.period = self.next_period
                self.next_period = DEFAULT_PERIOD_LENGTH

            # update local timer (can be made stochastic)

            # self.timer += rng.normal(1 + self.clock_drift_rate * 1e-3, self.clock_drift_scale) # FiGo EXTENSION
            self.timer += 1

            self.log_plot()
            self.log_epoch()

            yield self.env.timeout(1)

    def _tx(self, message):
        """Broadcast a *message* to all receivers."""

        if not self.neighbors:
            raise RuntimeError('There are no neighbors to send to.')

        for node in self.neighbors:
            # have a small probability of not sending the message
            # if random.random() < self.SEND_PROB:
            node.buffer.append(message)

    def log(self, *message):
        """Log a message with the current time and node id."""
        print('node', self.id, '|', 'time', self.env.now, '| epoch', self.epoch, '|', *message)

    def log_plot(self):
        node_phase[self.id][self.env.now] = self.timer

    def log_fire(self):
        node_fires[self.id][self.env.now] = 5

    def log_suppress(self):
        node_suppresses[self.id][self.env.now] = 1

    def log_epoch(self):
        node_epochs[self.id][self.env.now] = self.epoch


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

##########

# node configuration
init_state = [
    {'id': 0, 'initial_time_offset': 0, 'clock_drift_rate': 0, 'clock_drift_scale': 0.1, 'linestyle': 'solid',
     'hatch': '///'},
    {'id': 1, 'initial_time_offset': 10, 'clock_drift_rate': 5, 'clock_drift_scale': 0.1, 'linestyle': 'dashed',
     'hatch': '--'},
    {'id': 2, 'initial_time_offset': 20, 'clock_drift_rate': -5, 'clock_drift_scale': 0.1, 'linestyle': 'dotted',
     'hatch': '.'},
    {'id': 3, 'initial_time_offset': 30, 'clock_drift_rate': 5, 'clock_drift_scale': 0.1, 'linestyle': 'dashdot',
     'hatch': 'x'},
    {'id': 4, 'initial_time_offset': 40, 'clock_drift_rate': 5, 'clock_drift_scale': 0.1, 'linestyle': 'dashdot',
     'hatch': 'x'},
    # # {'id': 5, 'initial_time_offset': 50, 'clock_drift_rate': 0, 'clock_drift_scale': 0.1, 'linestyle': 'dashdot', 'hatch': 'x'},
    {'id': 5, 'initial_time_offset': 70, 'clock_drift_rate': 0, 'clock_drift_scale': 0.1, 'linestyle': 'dashdot',
     'hatch': 'x'},
]

# TOPO_FULLY_CONNECTED = {}

# add another 50 nodes to the fully connected topo and the init state variable
# for i in range(60):
#     TOPO_FULLY_CONNECTED[i] = [j for j in range(60) if j != i]
#     init_state.append(
#         {'id': i, 'initial_time_offset': random.randint(0, 99), 'clock_drift_rate': random.randint(-10, 10),
#          'clock_drift_scale': 0.1, 'linestyle': 'dashdot', 'hatch': 'x'})


topo = TOPO_FULLY_CONNECTED
# topo = TOPO_BRIDGE
DEFAULT_PERIOD_LENGTH = 100  # 1000 or 999? since 0 is included
SIM_TIME = 2000
MS_PROB = 1  # 0.8
RANDOM_SEED = 9

#######

# Set up data structures for logging
node_phase = [[0] * SIM_TIME for _ in init_state]
node_fires = [[0] * SIM_TIME for _ in init_state]
node_suppresses = [[0] * SIM_TIME for _ in init_state]
node_epochs = [[0] * SIM_TIME for _ in init_state]

random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

nodes = [PCONode5(env, state) for state in init_state]

for i, node in enumerate(nodes):
    node.neighbors = [nodes[n] for n in topo[i]]

env.run(until=SIM_TIME)

# Generate plots

# Generate graph plot
edge_list = [(kvs[0], v) for kvs in topo.items() for v in kvs[1]]
G = nx.from_edgelist(edge_list)
nx.draw(G, with_labels=True)

# Generate synchronization plots
fig, ax = plt.subplots(4, sharex=True)

# Node phase
for i in init_state:
    ax[0].plot(node_phase[i['id']], label='node ' + str(i['id']), linewidth=2,
               linestyle=i['linestyle'])  # , alpha=0.4) #
ax[0].set_title('Node phase')
ax[0].set(
    ylabel='Internal Timer Progress/Phase (ms)')  # xlabel='Time (ms)', ylabel='Phase (relative to default period length = 100s)')
ax[0].legend(loc="upper right")

# Fires and suppresses
ax[1].plot([sum(e) for e in zip(*node_fires)], color='red', label='node fire', linewidth=2)
ax[1].plot([sum(e) for e in zip(*node_suppresses)], color='grey', label='node supress', linewidth=2)
ax[1].set_title('Fires and suppresses')
ax[1].set(ylabel='Number of Fires / Suppresses')  # xlabel='Time (ms)',
ax[1].legend(loc="upper right")

# Node epoch
for i in init_state:
    ax[2].plot(node_epochs[i['id']], label='node ' + str(i['id']), linestyle=i['linestyle'], linewidth=2)
ax[2].set_title('Node epoch')
ax[2].set(ylabel='Local node epoch')
ax[2].legend(loc="upper right")

fig.suptitle('Initial time offsets: ' + str([i['initial_time_offset'] for i in init_state]))


def phase_diff(a, b):
    x = abs(a - b)
    return min(x, DEFAULT_PERIOD_LENGTH - x)


num_broadcasts = sum([sum([x != 0 for x in e]) for e in zip(*node_fires)])
range_list = [0] * SIM_TIME
stdev_list = [0] * SIM_TIME
for i, tick in enumerate(zip(*node_phase)):
    range_list[i] = (phase_diff(max(tick), min(tick))) / 2
    # stdev_list[i] = statistics.stdev(tick)
avg_range = round(sum(range_list) / len(range_list), 2)
# avg_stdev = sum(stdev_list) / len(stdev_list)

# Synchronization error
ax[3].plot(range_list, label='range', linewidth=2)
ax[3].set_title('Synchronization error (max-min)/2 (ticks)')
ax[3].set(xlabel='Time (ms)', ylabel='Synch error (ticks)')

# Print out metrics for the network
print('Number of broadcasts:', num_broadcasts)
print('Average synch range (ticks):', avg_range)
# print('Average synch st. dev. (ticks):', avg_stdev)
print("Avg. range after synchronization", sum(range_list[-100:]) / len(range_list[-100:]))

# plt.text(-1, -1, "Number of broadcasts: " + str(num_broadcasts))
plt.text(.80, .90, "Number of broadcasts: " + str(num_broadcasts), ha='left', va='top', transform=ax[3].transAxes)
plt.text(.80, .80, "Avg. synch range (ticks): " + str(avg_range), ha='left', va='top', transform=ax[3].transAxes)
# todo: add topo

plt.show()
