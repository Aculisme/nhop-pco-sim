from dataclasses import dataclass
from typing import Optional

import numpy as np  # todo: replace with cupy on linux?


class PCONode10:
    """Changing the notion of when messages are sent, so that information can be propagated throughout the period"""

    name = "FiGo Epochs Algorithm Version 10"

    def __init__(self, env, init_state, logging):  # , state):
        """Initialise the node with a given *env*, *initial state*, and *logging* handle."""

        self.env = env
        self.logging = logging

        self.id = init_state.id
        self.overall_mult = init_state.overall_mult
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
        self.neighbors = init_state.neighbors

        # set externally
        self.all_nodes = None

        # externally accessible
        self.buffer = []

        # used by main loop
        self.phase = 0
        self.epoch = 0
        # self.highest_msg_this_epoch = (self.id, self.epoch, self.phase)
        self.period = self.DEFAULT_PERIOD_LENGTH
        # self.super_out_of_sync = False
        # self.next_period = self.DEFAULT_PERIOD_LENGTH
        self.next_starting_phase = 0
        self.firing_phase = self.rng.integers(0, self.DEFAULT_PERIOD_LENGTH)
        self.fired = 0
        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting
        yield self.env.timeout(self.initial_time_offset * self.overall_mult)
        self.buffer.clear()

        while True:

            while self.buffer:

                new_msg = self.buffer.pop()
                msg_id, msg_epoch, msg_phase = new_msg

                phase_diff = (self.phase - msg_phase)

                current_epoch_phase_float = self.epoch + self.phase / self.period
                msg_epoch_phase_float = msg_epoch + msg_phase / self.period
                delta = current_epoch_phase_float - msg_epoch_phase_float

                if delta <= -1:  # todo: check < vs <=
                    # we're more than a period behind. We need to change both our epoch and phase to match theirs.

                    if phase_diff >= 0:  # todo: check > vs >=
                        # our phase is ahead of the message, so if we set our epoch to match theirs, they will then
                        # have to sync back to us (since we can't lengthen our next period...)
                        # instead, we should set our epoch to one less than theirs, and set our starting phase for the
                        # next period to
                        self.epoch = msg_epoch - 1
                    else:
                        # our phase is behind the message, so we can just set our epoch to match theirs and our phase to
                        # match theirs.
                        self.epoch = msg_epoch
                    self.next_starting_phase = abs(phase_diff)
                    self.log_fire_update()

                elif -1 < delta < 0:
                    # we're behind, but within a period. We need to change our phase to match theirs.

                    self.next_starting_phase = abs(phase_diff)
                    # todo: add logging support for phase synchronizations.

                elif delta >= 1:  # todo: check < vs <=
                    # the message received was more than an epoch behind, so they're out of synch.
                    # We should broadcast immediately to get them up to speed.

                    self._tx((self.id, self.epoch, self.phase))
                    self.log_out_of_sync_broadcast()

            # Firing phase reached, broadcast message
            if self.phase >= self.firing_phase and not self.fired:
                self.log('fired: broadcast')
                self._tx((self.id, self.epoch, self.phase))
                self.log_fire()
                self.fired = 1

            # timer expired
            if self.phase >= self.period:
                # increment epoch now that our timer has expired
                self.epoch += 1
                self.phase = self.next_starting_phase
                self.next_starting_phase = 0
                self.fired = 0

                # if no message seen this epoch, broadcast
                # if self.highest_msg_this_epoch[0] == self.id:
                #     self.tx((self.id, self.epoch, self.phase))
                #     self._log_fire()

                # else:
                #     self.tx((self.id, self.epoch, self.phase))
                #     self._log_fire()

                # 1-MS_PROB chance of overriding message suppression and firing anyway
                # elif self.rng.random() > self.MS_PROB:
                #     self.tx((self.id, self.epoch, self.phase))

                # we just heard a message with an epoch far ahead of our. We were out of sync!
                # So let everyone else know this as well.
                # elif self.super_out_of_sync:  # self.highest_msg_this_epoch[0] != self.id:
                #     self.log('triggered super out of sync broadcast')
                #     self.tx((self.id, self.epoch, self.phase))
                #     self.log_out_of_sync_broadcast()
                #     self.super_out_of_sync = False

                # suppress message
                # else:
                #     self.log_suppress()
                #     pass

                # self.highest_msg_this_epoch = (self.id, self.epoch, self.phase)

            self.log_phase()
            self.log_epoch()

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
        # self._log_fire()
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
            print('node', self.id, '|', 'time', self.env.now / self.overall_mult, '| phase', self.phase, '| epoch',
                  self.epoch, '|', *message)

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
            self.logging.node_phase_y[self.id].append(self.next_starting_phase)
            self.logging.node_phase_percentage_x[self.id].append(self.env.now)
            self.logging.node_phase_percentage_y[self.id].append(self.next_starting_phase / max(self.period, 1) * 100)

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
        self.logging.reception_y.append(self.all_nodes[neighbor].id)

    def log_fire_update(self):
        self.logging.fire_update_x.append(self.env.now)
        self.logging.fire_update_y.append(self.id)

    def log_synch_to_node(self, new_msg):
        self.log('synchronized to node', new_msg[0], 'which has epoch', new_msg[1], 'and phase', new_msg[2],
                 'setting next starting phase to', self.next_starting_phase)
        # todo: add logging support for synchronizations.

    def log_out_of_sync_broadcast(self):
        self.logging.out_of_sync_broadcast_x.append(self.env.now)
        self.logging.out_of_sync_broadcast_y.append(self.id)
