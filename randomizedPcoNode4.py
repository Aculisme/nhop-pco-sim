import numpy as np  # todo: replace with cupy on linux?


class RandomizedPCONode4:
    """
    Adding exponential backoff and message suppression to RandomizedPCONode3. (My algo, needs a snappier name)
    """

    name = "Modified Random-Phase PCO version 4"

    def __init__(self, env, init_state, logging):
        """Initialise the node with a given *env*, *initial state*, and *logging* handle."""

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
                                                  self.clock_drift_rate_offset_range)
        self.LOGGING_ON = init_state.LOGGING_ON
        self.neighbors = init_state.neighbors
        self.phase_diff_percentage_threshold = init_state.phase_diff_percentage_threshold

        # set externally post initialization
        self.all_nodes = None

        # externally accessible
        self.buffer = []

        # used for phase synchronization
        self.phase = 0
        self.period = self.DEFAULT_PERIOD_LENGTH

        # used for epoch synchronization
        self.epoch = 0

        # used for exp. backoff
        self.next_fire = self.rng.integers(0, self.period)
        self.firing_counter = 0
        self.firing_interval_high = 8 * self.period
        self.firing_interval_low = 1 * self.period
        self.firing_interval = self.firing_interval_low

        # used for message suppression
        self.c = 0
        self.k = 1

        # start the main loop
        self._main_loop = self.env.process(self._main())

    def _main(self):
        """Main loop for the node"""

        # sleep random time before starting and clear buffer of any messages that arrived during sleep
        # (artefact of simulation setup)
        yield self.env.timeout(self.initial_time_offset * self.overall_mult)
        self.buffer.clear()

        while True:

            # On message received
            while self.buffer:
                new_msg = self.buffer.pop()
                msg_phase, msg_epoch = new_msg

                phase_diff_percentage = (msg_phase - self.phase) / self.period

                if msg_epoch > self.epoch:
                    self.epoch = msg_epoch

                    # simulation artefact:
                    if self.phase > msg_phase:
                        self.log_phase_helper(msg_phase)

                    self.phase = msg_phase

                    self.firing_interval = self.firing_interval_low
                    self.next_fire = self.rng.integers(0, self.firing_interval)
                    self.firing_counter = 0

                    self.c = 0

                elif msg_epoch == self.epoch and msg_phase > self.phase:
                    self.phase = msg_phase

                    # we are more than 10% behind the message
                    if phase_diff_percentage >= self.phase_diff_percentage_threshold:
                        self.firing_interval = self.firing_interval_low
                        self.next_fire = self.rng.integers(0, self.firing_interval)
                        self.firing_counter = 0
                        self.c = 0

                    # we're close enough to be synced
                    else:
                        self.c += 1
                        pass

                # todo: make enabling this a hyper-param -- has anti-synergy with low k values
                # elif msg_epoch < self.epoch:
                #     # they're out of sync and we should fire to let them know.
                #     # self.firing_interval = self.firing_interval_low
                #     self.log('fired: update broadcast')
                #     self._tx((self.phase, self.epoch))
                #     self.log_fire_update()

                # we're ahead
                else:
                    pass

            # timer until next fire expired
            if self.firing_counter >= self.next_fire and self.c < self.k:
                self.log('fired: broadcast')
                self._tx((self.phase, self.epoch))
                self.log_fire()
                self.next_fire = self.rng.integers(0, self.firing_interval)  # todo: add intervals to this
                self.firing_counter = 0

            # epoch timer expired
            if self.phase >= self.period:
                self.phase = 0
                self.epoch += 1

                # double the interval on each epoch expiry
                self.firing_interval = min(2 * self.firing_interval, self.firing_interval_high)

                # simulation artefact
                self.log_phase_helper(0)

            self.log_epoch()
            self.log_phase()

            # local timer update (stochastic)
            tick_len = int(
                self.rng.normal(
                    1 * self.RECEPTION_LOOP_TICKS + self.clock_drift_rate * 3e-6 * self.RECEPTION_LOOP_TICKS,
                    self.clock_drift_scale * self.RECEPTION_LOOP_TICKS))

            self.phase += self.RECEPTION_LOOP_TICKS
            self.firing_counter += self.RECEPTION_LOOP_TICKS

            self._last_tick_len = tick_len
            yield self.env.timeout(tick_len)

    def _tx(self, message):
        """Broadcast a *message* to all receivers."""
        self.log_fire()
        if not self.neighbors:
            raise RuntimeError('There are no neighbors to send to.')

        # todo: make reception model more complex:
        #   make reception dependent signal to noise ratio, like (Schmidt et al.)
        #   Including distance / position, transmit power, and interference
        for neighbor in self.neighbors:
            self.all_nodes[neighbor].buffer.append(message)
            self.log_reception(neighbor)

    def log(self, *message):
        """Log a message with the current time and node id."""
        if self.LOGGING_ON:
            print('node', self.id, '|', 'time', self.env.now / self.overall_mult, '| epoch', self.epoch, '|', *message)

    def log_phase(self):
        self.logging.node_phase_x[self.id].append(self.env.now)
        self.logging.node_phase_y[self.id].append(self.phase)
        self.logging.node_phase_percentage_x[self.id].append(self.env.now)
        self.logging.node_phase_percentage_y[self.id].append((self.phase / max(self.period, 1)) * 100)

    def log_phase_helper(self, phase_val):
        """needed to instantly set next env tick phase to 0, otherwise waits until next large tick to set to zero,
        messing up the interpolation when graphing"""
        self.logging.node_phase_x[self.id].append(self.env.now - self._last_tick_len)
        self.logging.node_phase_y[self.id].append(phase_val)
        self.logging.node_phase_percentage_x[self.id].append(self.env.now - self._last_tick_len)
        self.logging.node_phase_percentage_y[self.id].append((phase_val / max(self.period, 1)) * 100)

    def log_fire(self):
        self.logging.fire_x.append(self.env.now)
        self.logging.fire_y.append(self.id)

    def log_fire_update(self):
        self.logging.fire_update_x.append(self.env.now)
        self.logging.fire_update_y.append(self.id)

    def log_epoch(self):
        self.logging.node_epochs_x[self.id].append(self.env.now)
        self.logging.node_epochs_y[self.id].append(self.epoch)

    def log_reception(self, neighbor):
        self.logging.reception_x.append(self.env.now)
        self.logging.reception_y.append(self.all_nodes[neighbor].id)
