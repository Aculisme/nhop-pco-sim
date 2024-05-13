import random
import matplotlib.pyplot as plt

import simpy


class PCONode2:
    def __init__(self, env, init_state):  # , state):
        """Initialise the node with a given *env* and *state*."""
        # set externally
        self.neighbors = None

        # externally accessible
        self.buffer = simpy.Store(env)

        self.id = init_state['id']
        self.state = init_state
        self.env = env
        # self.pos = init_state.pos

        self.period = DEFAULT_PERIOD_LENGTH
        self.next_period = DEFAULT_PERIOD_LENGTH
        # self.period_timer = None
        self.time_offset_from_env = 0
        self.epoch = 0

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting
        yield self.env.timeout(self.state['initial_time_offset'])

        self.time_offset_from_env = self.env.now

        # self.epoch = 0
        max_epochs = 100

        while self.epoch < max_epochs:
            self.log('entering epoch:', self.epoch, 'next period:', self.next_period)

            # wait for period to be finished (while listening for incoming messages)
            has_received = yield self.env.process(self._period_loop())

            if not has_received:
                # then broadcast
                self.log('broadcast')
                self._tx('hello from ' + str(self.id))
            else:
                self.log('NOT broadcast')

            self.time_offset_from_env = self.env.now

            self.epoch += 1

    def _period_loop(self):
        """Waits for 1 period while listening for incoming messages (handled independently of the period timer)"""

        self.period = self.next_period
        random_drift = random.randint(0, 4) - 2
        # random_drift = 0
        period_timer = self.env.timeout(self.period + random_drift)

        has_received = False
        timer_done = False

        while not timer_done:
            with self.buffer.get() as req:

                # wait for either the period to finish or a message to be received
                ret = yield period_timer | req

                # received a message: log it then continue waiting
                if not period_timer.processed:  # todo: could probably check the state of the get instead?
                    # self.log('message received:', ret[req], "is first message?", not has_received)

                    # get time since last fire
                    timer_progress = self.env.now - self.time_offset_from_env  # todo: this is so so ugly

                    # only want to sync to the first received message and only if it arrives
                    #   'before' we fire ([period/2, period]) -- not before ([0, period/2]) (inclusive?)
                    if not has_received and timer_progress > (DEFAULT_PERIOD_LENGTH / 2) - 1:
                        # e = 0.5

                        # self.next_period = DEFAULT_PERIOD_LENGTH - int(e * (timer_progress / 2))

                        self.next_period = timer_progress

                        has_received = True
                        # self.log("syncing to received message")

                # period finished
                else:
                    timer_done = True

                    node_fires[self.id][self.env.now] = 1 + self.id

                    if not has_received:
                        self.next_period = DEFAULT_PERIOD_LENGTH

        return has_received

    def _tx(self, message):
        """Broadcast a *message* to all receivers."""

        # self.log("broadcasting")

        if not self.neighbors:
            raise RuntimeError('There are no neighbors to send to.')

        events = [node.buffer.put(message) for node in self.neighbors]

        return self.env.all_of(events)

    def log(self, *message):
        """Log a message with the current time and node id."""
        print('node', self.id, '|', 'time', self.env.now, '|', *message)


# num_nodes = 3
env = simpy.Environment()

# init_state = [
#     {'id': 0, 'initial_time_offset': 0},
#     {'id': 1, 'initial_time_offset': 1},
#     {'id': 2, 'initial_time_offset': 501},
# ]

DEFAULT_PERIOD_LENGTH = 1000  # 1000 or 999? since 0 is included
SIM_TIME = 10000

init_state = [
    {'id': 0, 'initial_time_offset': 0},
    {'id': 1, 'initial_time_offset': 100},
    {'id': 2, 'initial_time_offset': 200},
    {'id': 3, 'initial_time_offset': 300},
    {'id': 4, 'initial_time_offset': 400},
]

random.seed(10)

nodes = [PCONode2(env, state) for state in init_state]
for i, node in enumerate(nodes):
    # for now, neighbours of a given node are all the other nodes
    node.neighbors = nodes[:i] + nodes[i + 1:]

node_fires = {i['id']: [0] * SIM_TIME for i in init_state}
env.run(until=SIM_TIME)

# matplotlib plot

for i in init_state:
    plt.plot(node_fires[i['id']], label='node ' + str(i['id']), linewidth=2)
# plt.plot(node_fires[0], label='node 0', linewidth=3)
# plt.plot(node_fires[1], label='node 1')

plt.legend()
plt.show()
