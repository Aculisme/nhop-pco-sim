import random
import matplotlib.pyplot as plt

import networkx as nx
import simpy
import statistics
import numpy as np


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
        self.highest_msg_seen_this_epoch = (self.id, self.epoch)
        self.period = DEFAULT_PERIOD_LENGTH
        self.next_period = DEFAULT_PERIOD_LENGTH
        self.timer = 0
        self.messages_seen_this_epoch = 0

        self.MS_PROB = MS_PROB
        self.clock_drift_rate = init_state['clock_drift_rate']
        self.clock_drift_scale = init_state['clock_drift_scale']

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting
        yield self.env.timeout(self.state['initial_time_offset'])

        self.epoch = 0
        self.highest_msg_seen_this_epoch = (self.id, self.epoch)

        while True:

            while self.buffer:

                new_msg = self.buffer.pop()
                self.log('received message from node', new_msg[0], 'with epoch', new_msg[1])

                # If the new msg epoch is greater than the largest epoch message seen this period previously, update
                #   and sync to it
                # This will happen when all nodes are synced, since epoch is incremented before sending
                if new_msg[1] > self.highest_msg_seen_this_epoch[1] and new_msg[1] > self.epoch:
                    self.highest_msg_seen_this_epoch = new_msg

                    # synchronize to it
                    self.next_period = DEFAULT_PERIOD_LENGTH - (self.period - self.timer) - 1
                    # next period should be the time since the todo
                    self.log('synchronized to node', new_msg[0], 'which has epoch', new_msg[1],
                             'setting next period to', self.next_period)

                if new_msg[1] == self.highest_msg_seen_this_epoch[1]:
                    # new msg has same epoch but message arrived later, ignore.
                    pass

                if new_msg[1] >= self.epoch + 2:
                    # we're super out of sync, match their epoch number
                    self.epoch = new_msg[1]

            # timer expired
            if self.timer >= self.period:

                # increment epoch now that our timer has expired
                self.epoch += 1

                # if no message seen this epoch, broadcast
                if self.highest_msg_seen_this_epoch[0] == self.id:

                    self.log('broadcast')
                    self._tx((self.id, self.epoch))

                # todo: chance of ignoring suppression, depending on randomness OR connectivity?
                elif random.random() > self.MS_PROB:
                    self.log('broadcast (MS override)')
                    self._tx((self.id, self.epoch))

                else:
                    # suppress message
                    self.log_suppress()
                    pass

                self.timer = 0
                self.out_of_sync = False
                self.period = self.next_period
                self.next_period = DEFAULT_PERIOD_LENGTH
                self.highest_msg_seen_this_epoch = (self.id, self.epoch)

            # update local timer (can be made stochastic)
            self.timer += rng.normal(1 + self.clock_drift_rate * 1e-6, 0.001)  # self.clock_drift_scale
            # self.timer += 1
            self.log_plot()
            self.log_epoch()
            yield self.env.timeout(1)

    def _tx(self, message):
        """Broadcast a *message* to all receivers."""
        self.log_fire()
        self.log('fired')
        if not self.neighbors:
            raise RuntimeError('There are no neighbors to send to.')

        for neighbor in self.neighbors:
            distance = self.distance_euc(self.pos, neighbor.pos) / M_TO_PX
            reception_prob = 1 / (1 + distance ** 2)
            print(reception_prob)

            # message reception probability proportional to inverse distance squared
            if random.random() < reception_prob:
                neighbor.buffer.append(message)

    def distance_euc(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

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

TOPO_MINI_FC = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1],
}

##########

# node configuration
init_state = [
    {'id': 0, 'initial_time_offset': 0, 'clock_drift_rate': random.randint(-100, 100), 'clock_drift_scale': 0.1,
     'linestyle': 'solid',
     'hatch': '///'},
    {'id': 1, 'initial_time_offset': 10, 'clock_drift_rate': random.randint(-100, 100), 'clock_drift_scale': 0.1,
     'linestyle': 'dashed',
     'hatch': '--'},
    {'id': 2, 'initial_time_offset': 20, 'clock_drift_rate': random.randint(-100, 100), 'clock_drift_scale': 0.1,
     'linestyle': 'dotted',
     'hatch': '.'},
    {'id': 3, 'initial_time_offset': 30, 'clock_drift_rate': random.randint(-100, 100), 'clock_drift_scale': 0.1,
     'linestyle': 'dashdot',
     'hatch': 'x'},
    {'id': 4, 'initial_time_offset': 40, 'clock_drift_rate': random.randint(-100, 100), 'clock_drift_scale': 0.1,
     'linestyle': 'dashdot',
     'hatch': 'x'},
    {'id': 5, 'initial_time_offset': 101, 'clock_drift_rate': random.randint(-100, 100), 'clock_drift_scale': 0.1,
     'linestyle': 'dashdot',
     'hatch': 'x'},
]

# TOPO_FULLY_CONNECTED = {}

# add another 50 nodes to the fully connected topo and the init state variable
# for i in range(50):
#     TOPO_FULLY_CONNECTED[i] = [j for j in range(50) if j != i]
#     init_state.append(
#         {'id': i, 'initial_time_offset': random.randint(0, 500), 'clock_drift_rate': random.randint(-10, 10),
#          'clock_drift_scale': 0.1, 'linestyle': 'dashdot', 'hatch': 'x'})


# topo = TOPO_FULLY_CONNECTED
topo = TOPO_BRIDGE
# topo = TOPO_MINI_FC
DEFAULT_PERIOD_LENGTH = 100  # 1000 or 999? since 0 is included
SIM_TIME = 2000
MS_PROB = 1
# MS_PROB = 0.8
# MS_PROB = 1
RANDOM_SEED = 10
M_TO_PX = 300
# M_TO_PX = 750

#######

# Generate graph plot and positions
edge_list = [(kvs[0], v) for kvs in topo.items() for v in kvs[1]]
G = nx.from_edgelist(edge_list)
pos = nx.nx_agraph.graphviz_layout(G)
nx.draw(G, with_labels=True, pos=pos)
print(pos)

# Set up data structures for logging
node_phase = [[0] * SIM_TIME for _ in init_state]
node_fires = [[0] * SIM_TIME for _ in init_state]
node_suppresses = [[0] * SIM_TIME for _ in init_state]
node_epochs = [[0] * SIM_TIME for _ in init_state]

random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

nodes = [PCONode7(env, state) for state in init_state]

for i, node in enumerate(nodes):
    node.neighbors = [nodes[n] for n in range(len(nodes)) if n != i] #[nodes[n] for n in topo[i]]
    node.pos = pos[i]

env.run(until=SIM_TIME)

# Generate plots

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
    # todo: not accurate, because the phase should really be expressed as a % of the current period, not the default
    #  period


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
