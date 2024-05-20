from dataclasses import dataclass
from typing import Optional

import numpy as np  # todo: replace with cupy on linux?


class PCONode8:
    """Adding connectivity estimation and dynamic parameter adjustment to the PCO model."""

    def __init__(self, env, init_state, logging):  # , state):
        """Initialise the node with a given *env*, *initial state*, and *logging* handle."""

        self.env = env
        self.logging = logging

        self.id = init_state.id
        self.OVERALL_MULT = init_state.OVERALL_MULT
        self.RECEPTION_LOOP_TICKS = init_state.RECEPTION_LOOP_TICKS
        self.DEFAULT_PERIOD_LENGTH = init_state.DEFAULT_PERIOD_LENGTH
        self.MS_PROB = init_state.MS_PROB
        self.M_TO_PX = init_state.M_TO_PX
        self.DISTANCE_EXPONENT = init_state.DISTANCE_EXPONENT

        self.clock_drift_scale = init_state.clock_drift_scale
        self.pos = init_state.pos
        self.rng = np.random.default_rng(init_state.rng_seed)
        self.min_initial_time_offset = init_state.min_initial_time_offset
        self.max_initial_time_offset = init_state.max_initial_time_offset
        self.clock_drift_rate_offset_range = init_state.clock_drift_rate_offset_range
        self.initial_time_offset = self.rng.integers(self.min_initial_time_offset,
                                                     self.max_initial_time_offset)
        self.clock_drift_rate = self.rng.integers(-self.clock_drift_rate_offset_range,
                                                  self.clock_drift_rate_offset_range)
        self.LOGGING_ON = init_state.LOGGING_ON

        # set externally
        self.neighbors = None

        # externally accessible
        self.buffer = []

        # used by main loop
        self.timer = 0
        self.epoch = 0
        self.highest_msg_this_epoch = (self.id, self.epoch)
        self.period = self.DEFAULT_PERIOD_LENGTH
        self.next_period = self.DEFAULT_PERIOD_LENGTH

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting
        yield self.env.timeout(self.initial_time_offset * self.OVERALL_MULT)

        while True:

            while self.buffer:

                new_msg = self.buffer.pop()

                # If the new msg epoch is greater than the largest epoch message seen this period previously, update
                #   and sync to it
                # This will happen when all nodes are synced, since epoch is incremented before sending
                if new_msg[1] > self.highest_msg_this_epoch[1]:
                    self.highest_msg_this_epoch = new_msg

                    # synchronize to it
                    self.next_period = self.DEFAULT_PERIOD_LENGTH - (self.period - self.timer)  # - 1*OVERALL_MULT
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
                elif self.rng.random() > self.MS_PROB:
                    self.log('fired: broadcast (MS override)')
                    self._tx((self.id, self.epoch))

                else:
                    # suppress message
                    self.log_suppress()
                    self.log('fired: suppressed')
                    pass

                self.timer = 0
                self.period = self.next_period
                self.next_period = self.DEFAULT_PERIOD_LENGTH
                self.highest_msg_this_epoch = (self.id, self.epoch)

            self.log_phase()
            self.log_epoch()

            # local timer update (stochastic)
            tick_len = int(
                self.rng.normal(
                    1 * self.RECEPTION_LOOP_TICKS + self.clock_drift_rate * 3e-6 * self.RECEPTION_LOOP_TICKS,
                    self.clock_drift_scale * self.RECEPTION_LOOP_TICKS))

            self.timer += self.RECEPTION_LOOP_TICKS
            self.log_phase_helper()

            yield self.env.timeout(tick_len)

    def _tx(self, message):
        """Broadcast a *message* to all receivers."""
        self.log_fire()
        if not self.neighbors:
            raise RuntimeError('There are no neighbors to send to.')

        for neighbor in self.neighbors:
            distance = distance_euc(self.pos, neighbor.pos) / self.M_TO_PX
            reception_prob = reception_probability(distance, self.DISTANCE_EXPONENT)
            # message reception probability proportional to inverse distance squared
            # if random.random() < self.NEIGHBOR_RECEPTION_PROB:
            if self.rng.random() < reception_prob:
                neighbor.buffer.append(message)
                self.log_reception(neighbor)

    def log(self, *message):
        """Log a message with the current time and node id."""
        if self.LOGGING_ON:
            print('node', self.id, '|', 'time', self.env.now / self.OVERALL_MULT, '| epoch', self.epoch, '|', *message)

    def log_phase(self):
        self.logging.node_phase_x[self.id].append(self.env.now)
        self.logging.node_phase_y[self.id].append(self.timer)
        self.logging.node_phase_percentage_x[self.id].append(self.env.now)
        self.logging.node_phase_percentage_y[self.id].append((self.timer / max(self.period, 1)) * 100)

    def log_phase_helper(self):
        """needed to instantly set next env tick phase to 0, otherwise waits until next large tick to set to zero,
        messing up the interpolation when graphing"""
        if self.timer >= self.period:
            self.logging.node_phase_x[self.id].append(self.env.now)
            self.logging.node_phase_y[self.id].append(0)
            self.logging.node_phase_percentage_x[self.id].append(self.env.now)
            self.logging.node_phase_percentage_y[self.id].append(0)

    def log_fire(self):
        self.logging.fire_x.append(self.env.now)
        self.logging.fire_y.append(self.id)

    def log_suppress(self):
        self.logging.suppress_x.append(self.env.now)
        self.logging.suppress_y.append(self.id)

    def log_epoch(self):
        self.logging.node_epochs_x[self.id].append(self.env.now)
        self.logging.node_epochs_y[self.id].append(self.epoch)

    def log_reception(self, neighbor):
        self.logging.reception_x.append(self.env.now)
        self.logging.reception_y.append(neighbor.id)


def distance_euc(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def reception_probability(distance, distance_exponent):
    return 1 / (1 + distance ** distance_exponent)


@dataclass
class Logging:
    node_phase_x: list
    node_phase_y: list
    node_phase_percentage_x: list
    node_phase_percentage_y: list
    node_epochs_x: list
    node_epochs_y: list
    fire_x: list
    fire_y: list
    suppress_y: list
    suppress_x: list
    reception_x: list
    reception_y: list


@dataclass
class InitState:
    id: int
    # initial_time_offset: int
    # clock_drift_rate: int
    clock_drift_scale: float
    min_initial_time_offset: int
    max_initial_time_offset: int
    clock_drift_rate_offset_range: int
    OVERALL_MULT: int
    RECEPTION_LOOP_TICKS: int
    DEFAULT_PERIOD_LENGTH: int
    M_TO_PX: int
    DISTANCE_EXPONENT: int
    MS_PROB: float
    pos: tuple
    LOGGING_ON: bool
    rng_seed: int


RESULTS_CSV_HEADER = ','.join([
    'num_nodes',
    'num_broadcasts',
    'avg_mean_phase_diff_after_synchronization',
    'avg_max_phase_diff_after_synchronization',
    'time_until_synchronization_human_readable',
    'rng_seed',
    'ms_prob'
])


@dataclass
class Results:
    num_nodes: int
    # todo: density: float
    num_broadcasts: int
    avg_mean_phase_diff_after_synchronization: float
    avg_max_phase_diff_after_synchronization: float
    time_until_synchronization_human_readable: float
    rng_seed: int
    ms_prob: float
    # time_until_synchronization: float
    # metrics: list

    def to_iterable(self):
        return [
            self.num_nodes,
            self.num_broadcasts,
            self.avg_mean_phase_diff_after_synchronization,
            self.avg_max_phase_diff_after_synchronization,
            self.time_until_synchronization_human_readable,
            self.rng_seed,
            self.ms_prob
        ]

    def to_csv(self):
        return ','.join([str(x) for x in self.to_iterable()])
