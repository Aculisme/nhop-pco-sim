import random
import matplotlib.pyplot as plt

import simpy


class PCONode3:
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

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting
        yield self.env.timeout(self.state['initial_time_offset'])
        self.timer = 0

        while True:

            self.transmit = False

            # self.log('entering epoch:', self.epoch, 'next period:', self.next_period)

            # update local timer (can be made stochastic)
            self.timer += 1

            # timer fired
            if self.timer >= self.period:
                self.log('timer expired')
                self.transmit = True

            # look at received messages, and decide whether to broadcast
            if self.buffer:  # max(self.buffer) > self.timer:
                # self.log('message received:', self.buffer)

                if self.timer > DEFAULT_PERIOD_LENGTH / 2:
                    # node is firing earlier than us, sync to it
                    self.next_period = DEFAULT_PERIOD_LENGTH - (self.period - self.timer)
                    self.log('setting next period to:', self.next_period, 'from node', self.buffer[0], "'s message")
                    self.transmit = False

                self.buffer.clear()

            # broadcast
            if self.timer >= self.period and self.transmit:
                self._tx((self.id, self.epoch))  # self.timer
                self.log_fire()
                self.transmit = False
            elif self.timer >= self.period:
                self.log_suppress()

            if self.timer >= self.period:
                self.timer = 0
                self.period = self.next_period
                self.next_period = DEFAULT_PERIOD_LENGTH

            self.log_plot()
            yield self.env.timeout(1)

    def _tx(self, message):
        """Broadcast a *message* to all receivers."""

        if not self.neighbors:
            raise RuntimeError('There are no neighbors to send to.')

        for node in self.neighbors:
            node.buffer.append(message)

    def log(self, *message):
        """Log a message with the current time and node id."""
        print('node', self.id, '|', 'time', self.env.now, '|', *message)

    def log_plot(self):
        node_phase[self.id][self.env.now] = self.timer

    def log_fire(self):
        node_fires[self.id][self.env.now] = 5

    def log_suppress(self):
        node_suppresses[self.id][self.env.now] = 2


env = simpy.Environment()

init_state = [
    {'id': 0, 'initial_time_offset': 0, 'linestyle': 'solid', 'hatch': '///'},
    # {'id': 1, 'initial_time_offset': 170},
    {'id': 1, 'initial_time_offset': 40, 'linestyle': 'dashed', 'hatch': '--'},
    {'id': 2, 'initial_time_offset': 65, 'linestyle': 'dotted', 'hatch': '.'},
    # {'id': 3, 'initial_time_offset': 90, 'linestyle': 'dashdot', 'hatch': 'x'},
]

DEFAULT_PERIOD_LENGTH = 100  # 1000 or 999? since 0 is included

SIM_TIME = 1000

# node_phase = {i['id']: [0] * SIM_TIME for i in init_state}
# node_phase = {i['id']: [] for i in init_state}
node_phase = [[0] * SIM_TIME for i in init_state]
node_fires = [[0] * SIM_TIME for i in init_state]
node_suppresses = [[0] * SIM_TIME for i in init_state]

random.seed(10)

nodes = [PCONode3(env, state) for state in init_state]
for i, node in enumerate(nodes):
    # for now, neighbours of a given node are all the other nodes
    node.neighbors = nodes[:i] + nodes[i + 1:]

env.run(until=SIM_TIME)

# matplotlib plot
plt.figure(0)
plt.subplot(211)
for i in init_state:
    plt.plot(node_phase[i['id']], label='node ' + str(i['id']), linewidth=2, linestyle=i['linestyle'])  # , alpha=0.4) #
    # plt.fill_between(range(SIM_TIME), node_phase[i['id']], alpha=0.4)#, hatch=i['hatch'])

# plt.fill_between(range(SIM_TIME), node_phase[0], alpha=0.5)

plt.legend(loc="upper right")

plt.subplot(212)
for i in init_state:
    plt.plot(node_fires[i['id']], color='red', linewidth=2)
    plt.plot(node_suppresses[i['id']], color='grey', linewidth=2)

plt.suptitle('Initial time offsets: ' + str([i['initial_time_offset'] for i in init_state]))

plt.show()
