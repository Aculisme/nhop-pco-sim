from dataclasses import dataclass
from typing import Optional

import numpy as np  # todo: replace with cupy on linux?

# from supervisor2 import distance_euc, reception_probability


class RandomizedPCONode1:
    """Adding connectivity estimation and dynamic parameter adjustment to the PCO model."""

    name = "Randomized Phase PCO (Schmidt et al.) version 1"

    def __init__(self, env, init_state, logging):  # , state):
        """Initialise the node with a given *env*, *initial state*, and *logging* handle."""

        self.env = env
        self.logging = logging

        self.id = init_state.id
        self.OVERALL_MULT = init_state.overall_mult
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
                                                  self.clock_drift_rate_offset_range) if init_state.clock_drift_rate_offset_range > 0 else 0
        self.LOGGING_ON = init_state.LOGGING_ON

        # set externally
        self.neighbors = init_state.neighbors

        self.all_nodes = None

        # externally accessible
        self.buffer = []

        # used by main loop
        self.phase = 0
        self.firing_phase = self.rng.integers(0, 100) * self.OVERALL_MULT
        self.period = self.DEFAULT_PERIOD_LENGTH
        self.fired = 0

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def phase_response_function(self, phase_diff):
        a = 0.5
        if phase_diff <= self.DEFAULT_PERIOD_LENGTH / 2:
            return (1 - a) * phase_diff
        else:
            return (1 - a) * phase_diff + a * self.DEFAULT_PERIOD_LENGTH

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting and clear buffer of any messages that arrived during sleep
        # (artefact of simulation setup)
        yield self.env.timeout(self.initial_time_offset * self.OVERALL_MULT)
        self.buffer.clear()

        while True:

            # On message received
            if self.buffer:
                new_msg = self.buffer.pop(0)  # get first message to arrive in buffer
                msg_firing_phase = new_msg  # [0]
                phase_diff = (self.phase - msg_firing_phase) % self.period
                phase_diff_adjusted = self.phase_response_function(phase_diff)
                self.phase = (phase_diff_adjusted + msg_firing_phase) % self.DEFAULT_PERIOD_LENGTH
                self.buffer.clear()
                # if self.firing_phase < self.phase:
                #     self.fired = 1

            # todo: when we adjust our phase, it could be greater than the firing phase, so we shouldn't fire right after

            # timer expired
            if self.phase >= self.firing_phase and not self.fired:
                # print(self.phase, self.firing_phase)
                self.log('fired: broadcast')
                self._tx(self.firing_phase)
                self.fired = 1

            if self.phase >= self.period:
                self.phase = 0
                self.fired = 0
                # self.firing_phase = self.rng.integers(1, 100) * self.OVERALL_MULT

            self.log_phase()

            # local timer update (stochastic)
            tick_len = int(
                self.rng.normal(
                    1 * self.RECEPTION_LOOP_TICKS + self.clock_drift_rate * 3e-6 * self.RECEPTION_LOOP_TICKS,
                    self.clock_drift_scale * self.RECEPTION_LOOP_TICKS))

            self.phase += self.RECEPTION_LOOP_TICKS
            self.log_phase_helper()

            yield self.env.timeout(tick_len)

    def _tx(self, message):
        """Broadcast a *message* to all receivers."""
        self.log_fire()
        if not self.neighbors:
            raise RuntimeError('There are no neighbors to send to.')

        for neighbor in self.neighbors:
            # distance = distance_euc(self.pos, neighbor.pos) / self.M_TO_PX
            # reception_prob = reception_probability(distance, self.DISTANCE_EXPONENT)
            # message reception probability proportional to inverse distance squared
            # if random.random() < self.NEIGHBOR_RECEPTION_PROB:
            # if self.rng.random() < reception_prob:
            if True:
                self.all_nodes[neighbor].buffer.append(message)
                self.log_reception(neighbor)

    def log(self, *message):
        """Log a message with the current time and node id."""
        if self.LOGGING_ON:
            # print('node', self.id, '|', 'time', self.env.now / self.OVERALL_MULT, '| epoch', self.epoch, '|', *message)
            print('node', self.id, '|', 'time', self.env.now / self.OVERALL_MULT, '|', *message)

    def log_phase(self):
        self.logging.node_phase_x[self.id].append(self.env.now)
        self.logging.node_phase_y[self.id].append(self.phase)
        self.logging.node_phase_percentage_x[self.id].append(self.env.now)
        self.logging.node_phase_percentage_y[self.id].append((self.phase / max(self.period, 1)) * 100)

    def log_phase_helper(self):
        """needed to instantly set next env tick phase to 0, otherwise waits until next large tick to set to zero,
        messing up the interpolation when graphing"""
        if self.phase >= self.period:
            self.logging.node_phase_x[self.id].append(self.env.now)
            self.logging.node_phase_y[self.id].append(0)
            self.logging.node_phase_percentage_x[self.id].append(self.env.now)
            self.logging.node_phase_percentage_y[self.id].append(0)

    def log_fire(self):
        self.logging.fire_x.append(self.env.now)
        self.logging.fire_y.append(self.id)

    # def log_suppress(self):
    #     self.logging.suppress_x.append(self.env.now)
    #     self.logging.suppress_y.append(self.id)

    # def log_epoch(self):
    #     self.logging.node_epochs_x[self.id].append(self.env.now)
    #     self.logging.node_epochs_y[self.id].append(self.epoch)

    def log_reception(self, neighbor):
        self.logging.reception_x.append(self.env.now)
        self.logging.reception_y.append(self.all_nodes[neighbor].id)