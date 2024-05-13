import random
import matplotlib.pyplot as plt

import networkx as nx
import simpy


class PCONode4:
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

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting
        yield self.env.timeout(self.state['initial_time_offset'])

        while True:

            # update local timer (can be made stochastic)
            self.timer += 1

            # look at received messages, and decide whether to broadcast
            if self.buffer:  # max(self.buffer) > self.timer:

                if self.buffer[0][1] > self.epoch:
                    self.next_period = DEFAULT_PERIOD_LENGTH - (self.period - self.timer)
                    self.log('case 0: setting next period to:', self.next_period, 'from node', self.buffer[0][0],
                             "'s message")
                    self.messages_seen_this_epoch += 1
                    self.epoch = self.buffer[0][1]

                elif self.timer > DEFAULT_PERIOD_LENGTH / 2 and self.buffer[0][1] >= self.epoch:
                    self.next_period = DEFAULT_PERIOD_LENGTH - (self.period - self.timer)
                    self.log('case 1: setting next period to:', self.next_period, 'from node', self.buffer[0][0],
                             "'s message")
                    self.messages_seen_this_epoch += 1
                    self.epoch = self.buffer[0][1]

                self.buffer.clear()

            # timer expired
            if self.timer >= self.period:

                # broadcast
                if self.messages_seen_this_epoch < 1:
                    self.log('broadcast')
                    self.epoch += 1
                    self._tx((self.id, self.epoch))

                    self.log_fire()
                    self.log('fired')

                # suppress message
                else:
                    self.log_suppress()

                # reset timer & prepare for next period
                self.timer = 0
                self.period = self.next_period
                self.next_period = DEFAULT_PERIOD_LENGTH

            self.log_plot()
            self.log_epoch()
            yield self.env.timeout(1)

    def _tx(self, message):
        """Broadcast a *message* to all receivers."""

        if not self.neighbors:
            raise RuntimeError('There are no neighbors to send to.')

        for node in self.neighbors:
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

# topo = TOPO_FULLY_CONNECTED
topo = TOPO_BRIDGE

# generate (flattened) edge list from topo dict
edge_list = [(kvs[0], v) for kvs in topo.items() for v in kvs[1]]
G = nx.from_edgelist(edge_list)

# show graph
nx.draw(G, with_labels=True)

# node configuration
init_state = [
    {'id': 0, 'initial_time_offset': 0, 'linestyle': 'solid', 'hatch': '///'},
    {'id': 1, 'initial_time_offset': 10, 'linestyle': 'dashed', 'hatch': '--'},
    {'id': 2, 'initial_time_offset': 20, 'linestyle': 'dotted', 'hatch': '.'},
    {'id': 3, 'initial_time_offset': 30, 'linestyle': 'dashdot', 'hatch': 'x'},
    {'id': 4, 'initial_time_offset': 40, 'linestyle': 'dashdot', 'hatch': 'x'},
    {'id': 5, 'initial_time_offset': 50, 'linestyle': 'dashdot', 'hatch': 'x'},
]

DEFAULT_PERIOD_LENGTH = 100  # 1000 or 999? since 0 is included

SIM_TIME = 1000

node_phase = [[0] * SIM_TIME for _ in init_state]
node_fires = [[0] * SIM_TIME for _ in init_state]
node_suppresses = [[0] * SIM_TIME for _ in init_state]
node_epochs = [[0] * SIM_TIME for _ in init_state]

# random.seed(10)

nodes = [PCONode4(env, state) for state in init_state]

for i, node in enumerate(nodes):
    # node.neighbors = nodes[:i] + nodes[i + 1:]
    node.neighbors = [nodes[n] for n in topo[i]]

env.run(until=SIM_TIME)

# Generate plots
plt.figure(1)
fig, ax = plt.subplots(3, sharex=True)

#
for i in init_state:
    ax[0].plot(node_phase[i['id']], label='node ' + str(i['id']), linewidth=2,
               linestyle=i['linestyle'])  # , alpha=0.4) #
ax[0].set_title('Node phase')
ax[0].set(
    ylabel='Internal Timer Progress/Phase (s)')  # xlabel='Time (s)', ylabel='Phase (relative to default period length = 100s)')
ax[0].legend(loc="upper right")

#
ax[1].plot([sum(e) for e in zip(*node_fires)], color='red', label='node fire', linewidth=2)
ax[1].plot([sum(e) for e in zip(*node_suppresses)], color='grey', label='node supress', linewidth=2)
ax[1].set_title('Fires and suppresses')
ax[1].set(ylabel='Number of Fires / Suppresses')  # xlabel='Time (s)',
ax[1].legend(loc="upper right")

#
for i in init_state:
    ax[2].plot(node_epochs[i['id']], label='node ' + str(i['id']), linestyle=i['linestyle'], linewidth=2)
ax[2].set_title('Node epoch')
ax[2].set(xlabel='Time (s)', ylabel='Local node epoch')
ax[2].legend(loc="upper right")

fig.suptitle('Initial time offsets: ' + str([i['initial_time_offset'] for i in init_state]))

plt.show()
