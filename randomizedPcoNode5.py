import numpy as np  # todo: replace with cupy on linux?


class RandomizedPCONode5:
    """
    Fixing +1 bug
    """

    name = "Modified Random-Phase PCO version 5"

    def __init__(self, env, init_state, logging):
        """Initialise the node with a given *env*, *initial state*, and *logging* handle."""

        # 1. Necessary for implementation of algorithm:

        self.env = env
        self.logging = logging

        # set initial state
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
                                                  self.clock_drift_rate_offset_range) if init_state.clock_drift_rate_offset_range > 0 else 0
        self.LOGGING_ON = init_state.LOGGING_ON
        self.neighbors = init_state.neighbors
        self.phase_diff_percentage_threshold = init_state.phase_diff_percentage_threshold

        # set externally post initialization
        self.all_nodes = None

        # externally accessible
        self.buffer = []

        # 2. Necessary for theoretical algorithm:

        # used for phase synchronization
        self.period = self.DEFAULT_PERIOD_LENGTH
        self.phase = 0

        # used for epoch synchronization
        self.epoch = 0

        # used for exp. backoff
        self.firing_interval_high = init_state.firing_interval_high
        self.firing_interval_low = init_state.firing_interval_low
        self.backoff_coeff = init_state.backoff_coeff
        self.firing_counter = 0
        self.firing_interval = self.firing_interval_low
        self.next_fire = self.rng.integers(0, self.firing_interval)

        # used for message suppression
        self.k = init_state.k
        self.c = 0

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting and clear buffer of any messages that arrived during sleep
        # (artefact of simulation setup)
        yield self.env.timeout(self.initial_time_offset * self.overall_mult)
        self.buffer.clear()

        while True:

            # MESSAGE_RECEIVED procedure
            while self.buffer:
                # self._log(self.firing_interval, self.firing_counter)
                new_msg = self.buffer.pop()
                msg_arrival_time, msg_phase, msg_epoch = new_msg

                actual_msg_phase = (msg_phase + (self.env.now - msg_arrival_time)) % self.period

                # self._log("my phase", self.phase, "actual phase", actual_msg_phase)

                # arrival_diff = self.env.now - msg_arrival_time  # simulation artefact
                phase_diff_percentage = (actual_msg_phase - self.phase) / self.period

                self._log("firing counter", self.firing_counter, "next fire", self.next_fire, "c", self.c, "k", self.k)

                if msg_epoch > self.epoch:
                    self.epoch = msg_epoch

                    self._log_phase_helper(msg_phase)  # simulation artefact

                    self.phase = actual_msg_phase

                    self.reset_firing_interval()

                elif msg_epoch == self.epoch and actual_msg_phase > self.phase:

                    self._log_phase_helper(msg_phase)  # simulation artefact

                    self.phase = actual_msg_phase

                    # we are more than x% behind the message
                    if phase_diff_percentage >= self.phase_diff_percentage_threshold:
                        # self._log("we're behind, resetting firing interval")
                        self.reset_firing_interval()

                    # we're close enough to be synced
                    else:
                        # self._log("our phase:", self.phase, "their phase:", actual_msg_phase)
                        # TODO! WHAT ABOUT CASE WHERE THE SAME NODE IS SPAMMING THE AIR, SO WE INCREMENT C A LOT PASSSING K
                        # TODO SO WE NEVER END UP FIRING...? MAYBE AN IMMEDIATE FIRE UPON OUTDATED INFO WOULD HELP?
                        self.c += 1
                        pass

                # # todo: experimental
                elif msg_epoch == self.epoch and actual_msg_phase < self.phase:

                    # hear a message more than x% behind
                    if abs(phase_diff_percentage) >= self.phase_diff_percentage_threshold:
                        # print("got here")
                        self.reset_firing_interval()
                        # self.firing_interval = self.firing_interval_low
                        # # self.next_fire = self.rng.integers(0, self.firing_interval)
                        # # self.firing_counter = 0
                        # self.c = 0

                        self._log_fire_update()

                # todo: make enabling this a hyper-param -- has anti-synergy with low k values
                # elif msg_epoch < self.epoch:
                #     # they're out of sync and we should fire to let them know.
                #     # self.firing_interval = self.firing_interval_low
                #     self.log('fired: update broadcast')
                #     self.tx()
                #     self._log_fire_update()

                # we're ahead
                else:
                    pass

            # NEXT_FIRE_TIMER_EXPIRE procedure
            if self.firing_counter >= self.next_fire:
                # if self.c < self.k:
                    # self._log("firing counter", self.firing_counter, "next fire", self.next_fire, "c", self.c, "k", self.k)
                # if not self.c < self.k:
                #     self._log_suppression()
                #     self._log('fired: broadcast')
                self.tx()
                self._log_fire()
                self.next_fire = self.rng.integers(0, self.firing_interval)  # todo: add intervals to this
                self.firing_counter = 0
                # else:
                #     self._log_suppression()
                    # self._log('fired: suppression')
                    # self._log_fire_update()
                    # self.next_fire = self.rng.integers(0, self.firing_interval)

            # EPOCH_TIMER_EXPIRE procedure
            if self.phase >= self.period:
                self.phase = 0
                self.epoch += 1

                # double the interval on each epoch expiry
                self.firing_interval = min(self.backoff_coeff * self.firing_interval, self.firing_interval_high)

                self._log_phase_helper(0)  # simulation artefact

            self._log_epoch()
            self._log_phase()

            # local timer update (stochastic)
            tick_len = int(
                self.rng.normal(
                    1 * self.RECEPTION_LOOP_TICKS + self.clock_drift_rate * 3e-6 * self.RECEPTION_LOOP_TICKS,
                    self.clock_drift_scale * self.RECEPTION_LOOP_TICKS))

            self._last_tick_len = tick_len  # simulation artefact
            yield self.env.timeout(tick_len)

            self.phase += self.RECEPTION_LOOP_TICKS
            self.firing_counter += self.RECEPTION_LOOP_TICKS

    def reset_firing_interval(self):
        self.firing_interval = self.firing_interval_low
        sample = self.rng.integers(0, self.firing_interval) + self.firing_counter
        if sample < self.next_fire:
            self.next_fire = sample
            # self.firing_counter = 0
        self.c = 0

    def tx(self):
        """Broadcast a *message* to all receivers."""
        self._log_fire()
        if not self.neighbors:
            return
            # raise RuntimeError('There are no neighbors to send to.')

        # todo: make reception model more complex:
        #   make reception dependent signal to noise ratio, like (Schmidt et al.)
        #   Including distance / position, transmit power, and interference
        for i in self.neighbors:
            neighbor = self.all_nodes[i]
            neighbor.buffer.append((self.env.now, self.phase, self.epoch))
            self._log_reception(neighbor)

    def _log(self, *message):
        """Log a message with the current time and node id."""
        if self.LOGGING_ON:
            print('node', self.id, '|', 'time', self.env.now / self.overall_mult, '| epoch', self.epoch, '|', *message)

    def _log_phase(self):
        self.logging.node_phase_x[self.id].append(self.env.now)
        self.logging.node_phase_y[self.id].append(self.phase)
        self.logging.node_phase_percentage_x[self.id].append(self.env.now)
        self.logging.node_phase_percentage_y[self.id].append((self.phase / max(self.period, 1)) * 100)

    def _log_phase_helper(self, phase_val):
        """needed to instantly set next env tick phase to 0, otherwise waits until next large tick to set to zero,
        messing up the interpolation when graphing"""
        self.logging.node_phase_x[self.id].append(self.env.now - self._last_tick_len)
        self.logging.node_phase_y[self.id].append(phase_val)
        self.logging.node_phase_percentage_x[self.id].append(self.env.now - self._last_tick_len)
        self.logging.node_phase_percentage_y[self.id].append((phase_val / max(self.period, 1)) * 100)

    def _log_fire(self):
        self.logging.fire_x.append(self.env.now)
        self.logging.fire_y.append(self.id)

    def _log_fire_update(self):
        self.logging.fire_update_x.append(self.env.now)
        self.logging.fire_update_y.append(self.id)

    def _log_epoch(self):
        self.logging.node_epochs_x[self.id].append(self.env.now)
        self.logging.node_epochs_y[self.id].append(self.epoch)

    def _log_reception(self, neighbor):
        self.logging.reception_x.append(self.env.now)
        self.logging.reception_y.append(neighbor.id)

    def _log_suppression(self):
        self.logging.suppress_x.append(self.env.now)
        self.logging.suppress_y.append(self.id)
