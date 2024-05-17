import math
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt

import networkx as nx
import simpy
import numpy as np  # todo: replace with cupy on linux?
from pprint import pprint as pp
from numpy.lib.stride_tricks import sliding_window_view

import multiprocessing as mp
import time


class PCONode7:
    """stable version of pco + epoch, with logging and plotting capabilities."""

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

        self.initial_time_offset = init_state.initial_time_offset
        self.clock_drift_rate = init_state.clock_drift_rate
        self.clock_drift_scale = init_state.clock_drift_scale
        self.pos = init_state.pos
        self.rng = init_state.rng
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
                # self.log('received message from node', new_msg[0], 'with epoch', new_msg[1])

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
                elif random.random() > self.MS_PROB:
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
            if random.random() < reception_prob:
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


def reception_probability(distance, DISTANCE_EXPONENT):
    return 1 / (1 + distance ** DISTANCE_EXPONENT)


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
    initial_time_offset: int
    clock_drift_rate: int
    clock_drift_scale: float
    OVERALL_MULT: int
    RECEPTION_LOOP_TICKS: int
    DEFAULT_PERIOD_LENGTH: int
    M_TO_PX: int
    DISTANCE_EXPONENT: int
    MS_PROB: float
    pos: tuple
    rng: np.random.Generator
    LOGGING_ON: bool


@dataclass
class TrialConfig:
    testing: bool
    logging_on: bool
    random_seed: int
    overall_mult: int
    reception_loop_ticks: int
    default_period_length: int
    sim_time: int
    ms_prob: float
    m_to_px: int
    distance_exponent: int
    clock_drift_rate_offset_range: int
    clock_drift_variability: float
    min_initial_time_offset: int
    max_initial_time_offset: int
    num_nodes: int
    sync_epsilon: int
    sync_num_ticks: float
    file_out: str
    num_trials: int
    # topo: nx.Graph


@dataclass
class Results:
    num_nodes: int
    # todo: density: float
    num_broadcasts: int
    avg_mean_phase_diff_after_synchronization: float
    avg_max_phase_diff_after_synchronization: float
    time_until_synchronization_human_readable: float
    # time_until_synchronization: float
    # metrics: list


def main(trial_config):
    a = time.time()

    # Generate graph plot and positions
    topo_gen = nx.random_internet_as_graph

    G = topo_gen(trial_config.num_nodes)  # topo_gen(15, 1)
    pos = nx.nx_agraph.graphviz_layout(G)
    random.seed(time.time())
    rng = np.random.default_rng(trial_config.random_seed)

    reception_probabilities = {i: {} for i in range(trial_config.num_nodes)}
    for x, i in enumerate(pos):
        for y, j in enumerate(pos):
            if i != j:
                reception_prob = reception_probability(distance_euc(pos[i], pos[j]) / trial_config.m_to_px,
                                                       trial_config.distance_exponent)
                reception_probabilities[x][y] = round(reception_prob, 3)

    # if LOGGING_ON:
    # print('reception_probabilities:')
    # pp(reception_probabilities)

    nx.draw(G, with_labels=True, pos=pos)

    # initial node states
    init_states = [
        InitState(
            id=i,
            initial_time_offset=random.randint(trial_config.min_initial_time_offset,
                                               trial_config.max_initial_time_offset),
            clock_drift_rate=random.randint(-trial_config.clock_drift_rate_offset_range,
                                            trial_config.clock_drift_rate_offset_range),
            clock_drift_scale=trial_config.clock_drift_variability,
            OVERALL_MULT=trial_config.overall_mult,
            RECEPTION_LOOP_TICKS=trial_config.reception_loop_ticks,
            DEFAULT_PERIOD_LENGTH=trial_config.default_period_length,
            M_TO_PX=trial_config.m_to_px,
            DISTANCE_EXPONENT=trial_config.distance_exponent,
            MS_PROB=trial_config.ms_prob,
            pos=pos[i],
            rng=rng,
            LOGGING_ON=trial_config.logging_on,
        )
        for i in range(trial_config.num_nodes)
    ]

    # Set up data structures for logging
    logging = Logging(
        node_phase_x=[[] for _ in range(trial_config.num_nodes)],
        node_phase_y=[[] for _ in range(trial_config.num_nodes)],
        node_phase_percentage_x=[[] for _ in range(trial_config.num_nodes)],
        node_phase_percentage_y=[[] for _ in range(trial_config.num_nodes)],
        node_epochs_x=[[] for _ in range(trial_config.num_nodes)],
        node_epochs_y=[[] for _ in range(trial_config.num_nodes)],
        fire_x=[],
        fire_y=[],
        suppress_y=[],
        suppress_x=[],
        reception_x=[],
        reception_y=[],
    )

    env = simpy.Environment()
    # Create Nodes
    nodes = [PCONode7(env, init_state, logging) for init_state in init_states]

    for i, node in enumerate(nodes):
        node.neighbors = [nodes[n] for n in range(len(nodes)) if n != i]  # [nodes[n] for n in topo[i]]
        node.pos = pos[i]

    env.run(until=trial_config.sim_time)

    num_broadcasts = len(logging.fire_x)

    def calculate_phase_difference(wave1, wave2, max_phase_percentage=100):
        diff = np.abs(wave1 - wave2)
        diff_opposite = abs(max_phase_percentage - diff)
        return np.minimum(diff, diff_opposite)

    x = np.linspace(0, trial_config.sim_time, int(trial_config.sim_time / trial_config.reception_loop_ticks))

    interpolated = np.array(
        [np.interp(x, logging.node_phase_percentage_x[i], logging.node_phase_percentage_y[i]) for i in
         range(len(logging.node_phase_percentage_y))])

    differences = []
    # todo: inefficient to do pairwise calcs, change if we're using huge topologies
    # O(n(n-1)/2) complexity :(
    for i in range(len(logging.node_phase_percentage_y)):
        for j in range(i + 1, len(logging.node_phase_percentage_y)):
            differences.append(calculate_phase_difference(interpolated[i], interpolated[j]))

    differences = np.array(differences)
    mean_differences = np.mean(differences, axis=0)
    max_differences = np.max(differences, axis=0)

    time_until_synchronization = trial_config.sim_time

    # Find time until synchronization
    """We consider the network synchronized when the maximum pair-wise phase difference between all nodes at a given tick 
    is less than 2% of the period length, and remains so for 95% of the subsequent 100 ticks"""
    v = sliding_window_view(max_differences, 1000)
    for i, window in enumerate(v):
        if (window < trial_config.sync_epsilon).sum() >= trial_config.sync_num_ticks * len(window):
            time_until_synchronization = i * trial_config.reception_loop_ticks
            break

    avg_mean_phase_diff_after_synchronization = np.round(np.mean(mean_differences[-100:]), 2)
    avg_max_phase_diff_after_synchronization = np.round(np.mean(max_differences[-100:]), 2)
    time_until_synchronization_human_readable = np.round(time_until_synchronization / 1000, 2)

    # Print out metrics for the network
    print('Number of broadcasts:', num_broadcasts)
    print("Synchronized avg. phase diff: ", avg_mean_phase_diff_after_synchronization)
    print("Time until synchronization: ", time_until_synchronization_human_readable)
    b = time.time()
    print('Processing took', round(b - a, 3), 'seconds')

    # Generate plots
    if trial_config.logging_on:
        fig, ax = plt.subplots(4, sharex=True)

        # Node phase
        for i in range(trial_config.num_nodes):
            ax[0].plot(logging.node_phase_x[i], logging.node_phase_y[i], label='node ' + str(i), linewidth=2,
                       linestyle='dashdot')
        ax[0].set_title('Node phase')
        ax[0].set(
            ylabel='Time since last fire')
        ax[0].legend(loc="upper right")

        # # Fires, suppresses, and receptions
        ax[1].plot(logging.reception_x, logging.reception_y, '*', color='blue', label='message reception', markersize=7,
                   alpha=0.3)
        ax[1].plot(logging.suppress_x, logging.suppress_y, 'x', color='grey', label='node suppress', markersize=5)
        ax[1].plot(logging.fire_x, logging.fire_y, 'o', color='red', label='node fire', markersize=5)

        ax[1].set_xticks(np.arange(0, trial_config.sim_time + 1, trial_config.default_period_length))
        ax[1].grid()
        ax[1].set_title('Fires and suppresses')
        ax[1].set(ylabel='Node ID')
        ax[1].legend(loc="upper right")

        # Node epochs
        for i in range(trial_config.num_nodes):
            ax[2].plot(logging.node_epochs_x[i], logging.node_epochs_y[i], label='node ' + str(i), linestyle='dashdot',
                       linewidth=2)
        ax[2].set_title('Node epoch')
        ax[2].set(ylabel='Node epoch')
        ax[2].legend(loc="upper right")

        fig.suptitle(
            'Modified PCO Node Simulation for a ' + topo_gen.__name__ + ' topology with ' + str(
                trial_config.num_nodes) + ' nodes')

        # Phase difference
        ax[3].plot(x, mean_differences, label='mean phase difference', linewidth=2)
        ax[3].plot(x, max_differences, label='max phase difference', linewidth=2)
        ax[3].set_title('Phase difference')
        ax[3].set(xlabel='Time (ms)', ylabel='Pair-wise phase difference (%)')
        ax[3].axvline(x=time_until_synchronization, color='blue', label='Synchronization time', linewidth=2,
                      linestyle='--',
                      alpha=0.5, marker='o')
        ax[3].legend(loc="upper right")

        # Metrics
        metrics = [
            "Number of broadcasts: " + str(num_broadcasts),
            "Time until synchronization: " + str(time_until_synchronization_human_readable),
            "Synchronized avg. phase diff: " + str(avg_mean_phase_diff_after_synchronization) + '%',
            "Topo: " + str(topo_gen.__name__),
            "Message Suppression prob.: " + str(trial_config.ms_prob * 100) + '%',
            "Non-adj. node comm. prob. params.: C=" + str(trial_config.m_to_px) + ' E=' + str(
                trial_config.distance_exponent),
        ]

        for i, metric_text in enumerate(metrics):
            plt.text(.50, .96 - .1 * i, metric_text, ha='left', va='top', transform=ax[3].transAxes)

        plt.show()

    # todo: calculate properties about the graph, like connectivity, average degree, etc.
    # todo: store adjacency matrix, etc. for later analysis
    return Results(
        num_nodes=trial_config.num_nodes,
        num_broadcasts=num_broadcasts,
        avg_mean_phase_diff_after_synchronization=avg_mean_phase_diff_after_synchronization,
        avg_max_phase_diff_after_synchronization=avg_max_phase_diff_after_synchronization,
        time_until_synchronization_human_readable=time_until_synchronization_human_readable,
    )


default_config = TrialConfig(
    testing=False,
    logging_on=False,
    random_seed=12,
    overall_mult=1000,
    reception_loop_ticks=100,
    default_period_length=100 * 1000,
    sim_time=2000 * 1000,
    ms_prob=1.0,
    m_to_px=90,
    distance_exponent=15,
    clock_drift_rate_offset_range=100,
    clock_drift_variability=0.05,
    min_initial_time_offset=0,
    max_initial_time_offset=100,
    num_nodes=10,
    sync_epsilon=2,
    sync_num_ticks=1.0,
    file_out='/Users/lucamehl/Downloads/nhop-pco-sim/temp.txt',
    num_trials=250,
    # topo=nx.random_internet_as_graph,
    # file_out='temp.txt'
)


def worker(i, config, q):
    print("running job", i, "of", config.num_trials)
    m = main(config)
    res_string = f"{m.num_nodes},{m.num_broadcasts},{m.avg_mean_phase_diff_after_synchronization},{m.avg_max_phase_diff_after_synchronization},{m.time_until_synchronization_human_readable}\n"
    q.put(res_string)
    return res_string


def listener(q, config):
    """listens for messages on the q, writes to file. """

    if not os.path.isfile(config.file_out):
        with open(config.file_out, 'w') as f:
            f.write(
                "num_nodes,num_broadcasts,avg_mean_phase_diff_after_synchronization,"
                "avg_max_phase_diff_after_synchronization,time_until_synchronization_human_readable\n")

    with open(config.file_out, 'a+') as f:
        while True:
            s = q.get()
            if s == 'kill':
                print("killed")
                break
            # s = f"{m.num_nodes},{m.num_broadcasts},{m.avg_mean_phase_diff_after_synchronization},
            # {m.avg_max_phase_diff_after_synchronization},{m.time_until_synchronization_human_readable}\n"
            print("wrote")
            f.write(s)
            f.flush()


def main_2():
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(7)  # mp.cpu_count() + 2

    # put listener to work first
    watcher = pool.apply_async(listener, (q, default_config))

    # fire off workers
    jobs = []
    for i in range(250):
        job = pool.apply_async(worker, (i, default_config, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


if __name__ == '__main__':
    main_2()
