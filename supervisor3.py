import os
from dataclasses import replace, dataclass
from pprint import pprint as pp
from typing import Callable, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import networkx as nx
import simpy
import numpy as np  # todo: replace with cupy on linux?
from numpy.lib.stride_tricks import sliding_window_view

import multiprocessing as mp
import time

from metaAnalysis import analyze_results
from pcoNode10 import PCONode10
from pcoNode11 import PCONode11
from pcoNode12 import PCONode12
from pcoNode13 import PCONode13
from pcoNode8 import PCONode8
from pcoNode9 import PCONode9
from randomizedPcoNode1 import RandomizedPCONode1
from randomizedPcoNode2 import RandomizedPCONode2
from randomizedPcoNode3 import RandomizedPCONode3
from randomizedPcoNode4 import RandomizedPCONode4
from randomizedPcoNode5 import RandomizedPCONode5

import pickle


@dataclass
class Logging:
    num_nodes: int
    node_phase_x: Optional[list] = None
    node_phase_y: Optional[list] = None
    node_phase_percentage_x: Optional[list] = None
    node_phase_percentage_y: Optional[list] = None
    node_epochs_x: Optional[list] = None
    node_epochs_y: Optional[list] = None
    fire_x: Optional[list] = None
    fire_y: Optional[list] = None
    suppress_y: Optional[list] = None
    suppress_x: Optional[list] = None
    reception_x: Optional[list] = None
    reception_y: Optional[list] = None
    fire_update_x: Optional[list] = None
    fire_update_y: Optional[list] = None
    out_of_sync_broadcast_x: Optional[list] = None
    out_of_sync_broadcast_y: Optional[list] = None

    def __post_init__(self):
        if self.node_phase_x is None:
            self.node_phase_x = [[] for _ in range(self.num_nodes)]
        if self.node_phase_y is None:
            self.node_phase_y = [[] for _ in range(self.num_nodes)]
        if self.node_phase_percentage_x is None:
            self.node_phase_percentage_x = [[] for _ in range(self.num_nodes)]
        if self.node_phase_percentage_y is None:
            self.node_phase_percentage_y = [[] for _ in range(self.num_nodes)]
        if self.node_epochs_x is None:
            self.node_epochs_x = [[] for _ in range(self.num_nodes)]
        if self.node_epochs_y is None:
            self.node_epochs_y = [[] for _ in range(self.num_nodes)]
        if self.fire_x is None:
            self.fire_x = []
        if self.fire_y is None:
            self.fire_y = []
        if self.suppress_y is None:
            self.suppress_y = []
        if self.suppress_x is None:
            self.suppress_x = []
        if self.reception_x is None:
            self.reception_x = []
        if self.reception_y is None:
            self.reception_y = []
        if self.fire_update_x is None:
            self.fire_update_x = []
        if self.fire_update_y is None:
            self.fire_update_y = []
        if self.out_of_sync_broadcast_x is None:
            self.out_of_sync_broadcast_x = []
        if self.out_of_sync_broadcast_y is None:
            self.out_of_sync_broadcast_y = []


@dataclass
class InitState:
    id: int
    clock_drift_scale: float
    min_initial_time_offset: int
    max_initial_time_offset: int
    clock_drift_rate_offset_range: int
    overall_mult: int
    RECEPTION_LOOP_TICKS: int
    DEFAULT_PERIOD_LENGTH: int
    M_TO_PX: int
    DISTANCE_EXPONENT: int
    MS_PROB: float
    pos: tuple
    LOGGING_ON: bool
    rng_seed: int
    neighbors: list
    phase_diff_percentage_threshold: float
    k: Optional[int] = None
    firing_interval_low: Optional[int] = None
    firing_interval_high: Optional[int] = None
    backoff_coeff: Optional[float] = None


@dataclass
class Results:
    # todo: density: float
    num_broadcasts: int
    avg_mean_phase_diff_after_synchronization: float
    avg_max_phase_diff_after_synchronization: float
    time_until_synchronization_human_readable: float
    rng_seed: int

    HEADER = ','.join([
        'num_broadcasts',
        'avg_mean_phase_diff_after_synchronization',
        'avg_max_phase_diff_after_synchronization',
        'time_until_synchronization_human_readable',
        'rng_seed'])

    def to_iterable(self):
        return [
            self.num_broadcasts,
            self.avg_mean_phase_diff_after_synchronization,
            self.avg_max_phase_diff_after_synchronization,
            self.time_until_synchronization_human_readable,
            self.rng_seed,
        ]

    def to_csv(self):
        return ','.join([str(x) for x in self.to_iterable()])


def edgelist_to_neighbors(edge_list):
    neighbors = {}
    for edge in edge_list:
        if edge[0] not in neighbors:
            neighbors[edge[0]] = []
        if edge[1] not in neighbors:
            neighbors[edge[1]] = []
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    return neighbors


def calculate_phase_difference(wave1, wave2, max_phase_percentage=100):
    diff = np.abs(wave1 - wave2)
    diff_opposite = abs(max_phase_percentage - diff)
    return np.minimum(diff, diff_opposite)


def distance_euc(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def reception_probability(distance, distance_exponent):
    return 1 / (1 + distance ** distance_exponent)


def run_trial(trial_config):
    a = time.time()

    # Generate graph plot and positions
    topo = trial_config.topo
    G = topo(**trial_config.topo_params)

    pos = nx.nx_agraph.graphviz_layout(G)

    rng_seed = time.time_ns()
    if trial_config.random_seed is not None:
        rng_seed = trial_config.random_seed

    # reception_probabilities = {i: {} for i in range(trial_config.num_nodes)}
    # for x, i in enumerate(pos):
    #     for y, j in enumerate(pos):
    #         if i != j:
    #             reception_prob = reception_probability(distance_euc(pos[i], pos[j]) / trial_config.m_to_px,
    #                                                    trial_config.distance_exponent)
    #             reception_probabilities[x][y] = round(reception_prob, 3)
    #
    # if not trial_config.testing:
    #     print('reception_probabilities:')
    #     pp(reception_probabilities)

    if trial_config.show_topo:
        nx.draw(G, with_labels=True, pos=pos)

    neighbors_dict = edgelist_to_neighbors(G.edges)

    # initial node states
    init_states = [
        InitState(
            id=i,
            clock_drift_scale=trial_config.clock_drift_variability,
            min_initial_time_offset=trial_config.min_initial_time_offset,
            max_initial_time_offset=trial_config.max_initial_time_offset,
            clock_drift_rate_offset_range=trial_config.clock_drift_rate_offset_range,
            overall_mult=trial_config.overall_mult,
            RECEPTION_LOOP_TICKS=trial_config.reception_loop_ticks,
            DEFAULT_PERIOD_LENGTH=trial_config.default_period_length,
            M_TO_PX=trial_config.m_to_px,
            DISTANCE_EXPONENT=trial_config.distance_exponent,
            MS_PROB=trial_config.ms_prob,
            pos=pos[i],
            rng_seed=rng_seed + i,
            k=trial_config.k,
            firing_interval_low=trial_config.firing_interval_low,
            firing_interval_high=trial_config.firing_interval_high,
            backoff_coeff=trial_config.backoff_coeff,
            LOGGING_ON=trial_config.logging_on,
            neighbors=neighbors_dict[i] if neighbors_dict else [],
            phase_diff_percentage_threshold=trial_config.phase_diff_percentage_threshold,
        )
        for i in range(trial_config.num_nodes)
    ]

    # Set up data structures for logging
    logging = Logging(num_nodes=trial_config.num_nodes)

    env = simpy.Environment()

    # Create Nodes
    nodes = [trial_config.pco_node(env, init_state, logging) for init_state in init_states]

    for i, node in enumerate(nodes):
        node.all_nodes = nodes  # [nodes[n] for n in topo[i]]
        # node.all_nodes = [nodes[n] for n in range(len(nodes)) if n != i]  # [nodes[n] for n in topo[i]]
        # node.pos = pos[i]

    env.run(until=trial_config.sim_time * trial_config.overall_mult)

    num_broadcasts = len(logging.fire_x)

    x = np.linspace(0, trial_config.sim_time * trial_config.overall_mult,
                    int(trial_config.sim_time * trial_config.overall_mult / trial_config.reception_loop_ticks))

    interpolated = np.array(
        [np.interp(x, logging.node_phase_percentage_x[i], logging.node_phase_percentage_y[i]) for i in
         range(len(logging.node_phase_percentage_y))])
    # todo: can probably replace the ints of form len(logging.node_phase_percentage_y))

    differences = []
    # todo: inefficient to do pairwise calcs, change if we're using huge topologies
    # O(n(n-1)/2) complexity :(
    for i in range(len(logging.node_phase_percentage_y)):
        for j in range(i + 1, len(logging.node_phase_percentage_y)):
            differences.append(calculate_phase_difference(interpolated[i], interpolated[j]))

    differences = np.array(differences)
    mean_differences = np.mean(differences, axis=0)
    max_differences = np.max(differences, axis=0)

    time_until_synchronization = trial_config.sim_time * trial_config.overall_mult

    all_nodes_start_time = [node.initial_time_offset * (trial_config.overall_mult / trial_config.reception_loop_ticks) for node in nodes]
    print(all_nodes_start_time)
    earliest_possible_sync_time = max(all_nodes_start_time)

    # Find time until `synchronization
    """We consider the network synchronized when the maximum pair-wise phase difference between all nodes at a given tick 
    is less than 2% of the period length, and remains so for 100% of the subsequent 100 ticks"""
    v = sliding_window_view(max_differences, 1000)
    for i, window in enumerate(v):
        if i < earliest_possible_sync_time:
            continue

        if (window < trial_config.sync_epsilon).sum() >= trial_config.sync_num_ticks * len(window):
            time_until_synchronization = i * trial_config.reception_loop_ticks
            break

    avg_mean_phase_diff_after_synchronization = np.round(np.mean(mean_differences[-100:]), 2)
    avg_max_phase_diff_after_synchronization = np.round(np.mean(max_differences[-100:]), 2)
    time_until_synchronization_human_readable = np.round(time_until_synchronization / 1000, 2)

    # Print out metrics for the network
    b = time.time()
    print('processing time', round(b - a, 3), 's', '|',
          '#Nodes:', trial_config.num_nodes, '|',
          'TTS:', time_until_synchronization_human_readable, 's', '|',
          '#Broadcasts:', num_broadcasts, '|',
          'Synched `max phase diff:', avg_max_phase_diff_after_synchronization, '%')

    # Generate plots
    if not trial_config.testing:
        fig, ax = plt.subplots(4, sharex=True)

        # Node phase
        for i in range(trial_config.num_nodes):
            ax[0].plot(logging.node_phase_x[i], logging.node_phase_y[i], label='node ' + str(i),
                       linewidth=2,
                       linestyle='dashdot')
        # ax[0].set_title('Node phase')
        # ax[0].set(
        #     ylabel='Phase')
        ax[0].set_title('Phase (us)')
        # ax[1].set(xlabel='Time (us)')
        # ax[0].legend(loc="upper right")

        # Fires, suppresses, and receptions
        ax[1].plot(logging.reception_x, logging.reception_y, '*', color='blue', label='message reception', markersize=7,
                   alpha=0.3)
        ax[1].plot(logging.suppress_x, logging.suppress_y, 'x', color='grey', label='node suppress', markersize=5)
        ax[1].plot(logging.fire_x, logging.fire_y, 'o', color='red', label='node fire', markersize=5)
        ax[1].plot(logging.fire_update_x, logging.fire_update_y, 'o', color='green', label='node update fire',
                   markersize=5)
        ax[1].plot(logging.out_of_sync_broadcast_x, logging.out_of_sync_broadcast_y, 'o', color='purple',
                   label='node out of sync broadcast',
                   markersize=5)

        # ax[1].set_xticks(
        #     np.arange(0, trial_config.sim_time * trial_config.overall_mult + 1, trial_config.default_period_length))
        # ax[1].grid()
        # ax[1].set_title('Fires and suppresses')
        # ax[1].set(ylabel='Node ID')
        # ax[1].legend(loc="upper right")

        # Node epochs
        for i in range(trial_config.num_nodes):
            ax[2].plot(logging.node_epochs_x[i], logging.node_epochs_y[i], label='node ' + str(i), linestyle='dashdot',
                       linewidth=2)
        ax[2].set_title('Epoch')
        ax[2].set(ylabel='Epoch')
        ax[2].legend(loc="upper right")

        # fig.suptitle(
        #     trial_config.pco_node.name + ' Node Simulation for a ' + topo.__name__ + ' topology with ' + str(
        #         trial_config.num_nodes) + ' nodes')
        #

        # Phase difference
        ax[3].plot(x, mean_differences, label='mean phase difference', linewidth=2)
        ax[3].plot(x, max_differences, label='max phase difference', linewidth=2)
        ax[3].set_title('Phase difference')
        ax[3].set(xlabel='Time (ms)', ylabel='Pair-wise phase difference (%)')
        ax[3].axvline(x=time_until_synchronization, color='blue', label='Synchronization time', linewidth=2,
                      linestyle='--',
                      alpha=0.5, marker='o')
        ax[3].legend(loc="upper right")

        # Node phase tests
        # for i in range(trial_config.num_nodes):
        #     x = logging.node_phase_x[i]
        #     y = logging.node_phase_y[i]
        #     colors = cm.viridis(np.linspace(0, 1, max(y)+1)) # , len(ys)
        #     mycols = [colors[int(j)] for j in y]
        #     # for y, c in zip(ys, colors):
        #     ax[4].scatter(x, [i]*len(x), color=mycols)
        #     # ax[4].plot(logging.node_phase_x[0], [0]*len(logging.node_phase_x[0]), color=colors, label='node ' + str(0),
        #     #            linewidth=(3 if 0 == 0 else 2),
        #     #            linestyle='dashdot')
        # ax[4].set_title('Node phase 2')
        # ax[4].set(
        #     ylabel='Time since last fire')
        # ax[4].legend(loc="upper right")

        # Metrics
        metrics = [
            "Number of broadcasts: " + str(num_broadcasts),
            "Time until synchronization: " + str(time_until_synchronization_human_readable),
            "Synchronized avg. phase diff: " + str(avg_mean_phase_diff_after_synchronization) + '%',
            "Topo: " + str(topo.__name__),
            "Message Suppression prob.: " + str(trial_config.ms_prob * 100) + '%',
            "Non-adj. node comm. prob. params.: C=" + str(trial_config.m_to_px) + ' E=' + str(
                trial_config.distance_exponent),
            "Random seed: " + str(rng_seed),
            "Phase difference threshold for cancelling exp. backoff: " + str(
                trial_config.phase_diff_percentage_threshold * 100) + "%",
            "k (message suppression threshold): " + str(trial_config.k),
        ]

        for i, metric_text in enumerate(metrics):
            plt.text(.50, .96 - .1 * i, metric_text, ha='left', va='top', transform=ax[3].transAxes)

        plt.show()

    # todo: calculate properties about the graph, like connectivity, average degree, etc.
    # todo: store adjacency matrix, etc. for later analysis in different file?
    return Results(
        num_broadcasts=num_broadcasts,
        avg_mean_phase_diff_after_synchronization=avg_mean_phase_diff_after_synchronization,
        avg_max_phase_diff_after_synchronization=avg_max_phase_diff_after_synchronization if trial_config.num_nodes > 1 else 0,
        time_until_synchronization_human_readable=time_until_synchronization_human_readable,
        rng_seed=rng_seed,
    )


@dataclass
class TrialConfig:
    # todo: set defaults?
    testing: bool
    logging_on: bool
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
    sync_epsilon: float  # threshold for synchronization (max pair-wise phase difference % of period length)
    sync_num_ticks: int  # number of ticks to be synchronized ?
    file_out: str
    num_trials: int
    phase_diff_percentage_threshold: float
    pco_node: Callable
    topo: Callable  # nx.Graph generator function
    show_topo: bool
    k: Optional[int] = None
    firing_interval_low: Optional[int] = None
    firing_interval_high: Optional[int] = None
    backoff_coeff: Optional[float] = None
    topo_params: Optional[dict] = None  # param : value, passed to generator
    # Setting random_seed will fix all trials to use the exact same random seed. If argument not passed, then each trial
    # will use a different time-based random seed.
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.topo_params is None:
            self.topo_params = {'n': self.num_nodes}
        # if self.random_seed is not None:
        #     self.topo_params['seed'] = self.random_seed


@dataclass
class AlgoConfig:
    algo_name: str
    default_period_length: int
    clock_drift_rate_offset_range: int
    clock_drift_variability: float
    min_initial_time_offset: int
    max_initial_time_offset: int
    pco_node: Callable
    ms_prob: Optional[float] = 0.0
    m_to_px: Optional[int] = 90
    distance_exponent: Optional[int] = 15
    phase_diff_percentage_threshold: Optional[float] = 2.0
    k: Optional[int] = 2
    firing_interval_low: Optional[int] = 1 * 100 * 1000
    firing_interval_high: Optional[int] = 8 * 100 * 1000
    backoff_coeff: Optional[float] = 2

    HEADER = ','.join([
        'algo_name',
        'default_period_length',
        'clock_drift_rate_offset_range',
        'clock_drift_variability',
        'min_initial_time_offset',
        'max_initial_time_offset',
        'ms_prob',
        'm_to_px',
        'distance_exponent',
        'phase_diff_percentage_threshold',
        'k',
        'firing_interval_low',
        'firing_interval_high',
        'backoff_coeff',
    ])

    def to_csv(self):
        return ','.join([str(x) for x in [
            self.algo_name,
            self.default_period_length,
            self.clock_drift_rate_offset_range,
            self.clock_drift_variability,
            self.min_initial_time_offset,
            self.max_initial_time_offset,
            self.ms_prob,
            self.m_to_px,
            self.distance_exponent,
            self.phase_diff_percentage_threshold,
            self.k,
            self.firing_interval_low,
            self.firing_interval_high,
            self.backoff_coeff,
        ]])


@dataclass
class TopoConfig:
    topo_name: str
    num_nodes: int
    topo_params: dict
    topo: Callable
    neighbors_dict: Optional[dict] = None

    HEADER = ','.join([
        'topo_name',
        'num_nodes',
        # 'topo_params',
        # 'neighbors_dict',
    ])

    def __post_init__(self):
        if self.neighbors_dict is None:
            self.neighbors_dict = edgelist_to_neighbors(self.topo(**self.topo_params).edges)

    def to_csv(self):
        return ','.join([str(x) for x in [
            self.topo_name,
            self.num_nodes,
            # self.topo_params,
            # self.neighbors_dict,
        ]])


@dataclass
class TestConfig:
    sim_time: int
    file_out: str
    num_trials: int

    HEADER = ','.join([
        'sim_time',
        'num_trials',
    ])

    def to_csv(self):
        return ','.join([str(x) for x in [
            self.sim_time,
            self.num_trials,
        ]])


def writer(test_config, results_queue):
    """listens for messages on the q, writes to file. """

    # create file with header if it doesn't exist
    if not os.path.isfile(test_config.file_out):
        with open(test_config.file_out, 'w+') as f:
            f.write(','.join([TestConfig.HEADER, AlgoConfig.HEADER, TopoConfig.HEADER, Results.HEADER]) + "\n")

    with open(test_config.file_out, 'a+') as f:
        while True:
            s = results_queue.get()
            if s == 'kill':
                print("done writing")
                break
            f.write(s + '\n')
            f.flush()


def worker(i, test_config, topo_config, algo_config, results_queue):
    # trial_config = replace(randomizedPcoNode5_config,
    #                        num_nodes=topo_config.num_nodes,
    #                        topo=topo_config.topo,
    #                        topo_params=topo_config.topo_params,
    #                        pco_node=algo_config.pco_node,
    #                        default_period_length=algo_config.default_period_length,
    #                        clock_drift_rate_offset_range=algo_config.clock_drift_rate_offset_range,
    #                        clock_drift_variability=algo_config.clock_drift_variability,
    #                        min_initial_time_offset=algo_config.min_initial_time_offset,
    #                        max_initial_time_offset=algo_config.max_initial_time_offset,
    #                        ms_prob=algo_config.ms_prob,
    #                        m_to_px=algo_config.m_to_px,
    #                        distance_exponent=algo_config.distance_exponent,
    #                        phase_diff_percentage_threshold=algo_config.phase_diff_percentage_threshold,
    #                        k=algo_config.k,
    #                        firing_interval_low=algo_config.firing_interval_low,
    #                        firing_interval_high=algo_config.firing_interval_high,
    #                        backoff_coeff=algo_config.backoff_coeff,
    #                        random_seed=None,
    #                        num_trials=1,
    #                        file_out=test_config.file_out
    #                        )

    # initialise TrialConfig with values from test_config, topo_config, and algo_config
    trial_config = TrialConfig(
        testing=True,
        logging_on=False,
        show_topo=False,
        overall_mult=1000,
        reception_loop_ticks=100,
        default_period_length=100 * 1000,
        sim_time=test_config.sim_time,  # * 1000,
        ms_prob=algo_config.ms_prob,  # 1.0,  # todo: make optional?
        m_to_px=algo_config.m_to_px,
        distance_exponent=algo_config.distance_exponent,
        clock_drift_rate_offset_range=algo_config.clock_drift_rate_offset_range,  # 50,  # 100,
        clock_drift_variability=algo_config.clock_drift_variability,  # 0.05,
        min_initial_time_offset=algo_config.min_initial_time_offset,
        max_initial_time_offset=algo_config.max_initial_time_offset,
        sync_epsilon=2,
        sync_num_ticks=1,
        file_out=test_config.file_out,
        num_trials=test_config.num_trials,
        phase_diff_percentage_threshold=algo_config.phase_diff_percentage_threshold,
        k=algo_config.k,
        backoff_coeff=algo_config.backoff_coeff,
        firing_interval_low=algo_config.firing_interval_low,
        firing_interval_high=algo_config.firing_interval_high,
        num_nodes=topo_config.num_nodes,
        pco_node=algo_config.pco_node,
        topo=topo_config.topo,
        topo_params=topo_config.topo_params,
        random_seed=None,
    )

    benchmark_results = run_trial(trial_config)
    res_string = ','.join(
        [test_config.to_csv(), algo_config.to_csv(), topo_config.to_csv(), benchmark_results.to_csv()])
    # plt.show()
    results_queue.put(res_string)
    return res_string


def run_test(topo_configs, algo_configs, test_config):
    t1 = time.time()
    manager = mp.Manager()
    results_queue = manager.Queue()
    pool = mp.Pool(7)  # mp.cpu_count() + 2

    w = pool.apply_async(writer, (test_config, results_queue))

    jobs = []

    for topo_config in topo_configs:
        for algo_config in algo_configs:
            for i in range(test_config.num_trials):
                job = pool.apply_async(worker, (i, test_config, topo_config, algo_config, results_queue))
                jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # now we are done, kill the writer
    results_queue.put('kill')
    pool.close()
    pool.join()

    print('Done. Completed in', round(time.time() - t1, 2), 'seconds')


randomizedPcoNode5_config = TrialConfig(
    testing=False,
    logging_on=True,
    show_topo=True,
    overall_mult=1000,
    reception_loop_ticks=100,
    default_period_length=100 * 1000,
    sim_time=1000,  # * 1000,
    ms_prob=0.0,  # 1.0,  # todo: make optional?
    m_to_px=90,
    distance_exponent=15,
    clock_drift_rate_offset_range=0,  # 50,  # 100,
    clock_drift_variability=0.05,  # 0.05,
    min_initial_time_offset=0,
    max_initial_time_offset=199,
    sync_epsilon=2,
    sync_num_ticks=1,
    # todo: make file path dynamic? or not?
    file_out='/Users/lucamehl/Downloads/nhop-pco-sim/randomized_pco.txt',
    num_trials=1,
    phase_diff_percentage_threshold=0.005,
    k=200,
    backoff_coeff=2,
    firing_interval_low=0.5 * (100 * 1000),
    firing_interval_high=20 * (100 * 1000),
    # num_nodes=5,
    num_nodes=11,
    # pco_node=RandomizedPCONode1,
    pco_node=RandomizedPCONode5,
    # pco_node=PCONode8,
    # topo=nx.path_graph,
    # topo=nx.cycle_graph,
    topo=nx.barbell_graph,  # todo: note: num_nodes = m1*2 + m2
    # topo=nx.complete_graph,
    # topo=nx.random_internet_as_graph,
    random_seed=2,
    # random_seed=1716582139984742000,
    # random_seed=1716631278258634000,
    # random_seed=1716676895879946000,
    topo_params={'m1':  5, 'm2': 1},  # seed
    # topo_params={'radius': 5, 'n': 32},  # seed
    # topo_params={'n': 16},
)

if __name__ == '__main__':

    # OPTION 1: RUN A SINGLE TRIAL WITH A SINGLE TOPOLOGY AND ALGORITHM
    run_trial(randomizedPcoNode5_config)

    # OPTION 2: RUN A TEST WITH MULTIPLE TOPOLOGIES AND ALGORITHMS

    # topo_configs = [
    #     TopoConfig(
    #         topo_name='linear',
    #         topo_params={'n': i},
    #         topo=nx.path_graph,
    #         num_nodes=i,
    #     ) for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]  # 9] # +[i for i in range(10, 128, 5)]
    # ]  # + [
    # #     TopoConfig(
    # #         topo_name='barbell',
    # #         topo_params={'m1': i, 'm2': 1},
    # #         topo=nx.barbell_graph,
    # #         num_nodes=2 * i + 1,
    # #     ) for i in [3, 4, 5, 6, 7, 8, 9] + [i for i in range(10, 50, 10)]
    # # ]
    #
    # test_config = TestConfig(
    #     sim_time=8000,
    #     num_trials=100,  # 3,
    #     file_out='out/test6.csv',  # /Users/lucamehl/Downloads/nhop-pco-sim/
    # )
    #
    # algo_configs = [
    #     AlgoConfig(
    #         algo_name='RP5',
    #         pco_node=RandomizedPCONode5,
    #         default_period_length=100 * 1000,
    #         clock_drift_rate_offset_range=0,
    #         clock_drift_variability=0.05,
    #         min_initial_time_offset=0,
    #         max_initial_time_offset=199,
    #         phase_diff_percentage_threshold=0.02,
    #         k=2,
    #         backoff_coeff=2,
    #         firing_interval_low=1 * (100 * 1000),
    #         firing_interval_high=16 * (100 * 1000),
    #     ),
    #     AlgoConfig(
    #         algo_name='PCO8_full',
    #         default_period_length=100 * 1000,
    #         ms_prob=0.0,
    #         clock_drift_rate_offset_range=0,
    #         clock_drift_variability=0.05,
    #         min_initial_time_offset=0,
    #         max_initial_time_offset=199,
    #         pco_node=PCONode8,
    #     ),
    #     AlgoConfig(
    #         algo_name='RP1',
    #         pco_node=RandomizedPCONode1,
    #         default_period_length=100 * 1000,
    #         clock_drift_rate_offset_range=0,
    #         clock_drift_variability=0.05,
    #         min_initial_time_offset=0,
    #         max_initial_time_offset=199,
    #         # phase_diff_percentage_threshold=0.02,
    #         # k=2,
    #         # backoff_coeff=2,
    #         # firing_interval_low=1 * (100 * 1000),
    #         # firing_interval_high=16 * (100 * 1000),
    #     ),
    #     AlgoConfig(
    #         algo_name='PCO8_ms',
    #         default_period_length=100 * 1000,
    #         ms_prob=1.0,
    #         clock_drift_rate_offset_range=0,
    #         clock_drift_variability=0.05,
    #         min_initial_time_offset=0,
    #         max_initial_time_offset=199,
    #         pco_node=PCONode8,
    #     )
    # ]

    # run_test(topo_configs, algo_configs, test_config)
    # analyze_results(test_config, topo_configs, algo_configs)

    # For option 2, use the following to save the topologies if needed
    # PIK = "topo_configs/linear_topo_configs_v1.dat"
    # with open(PIK, "wb") as f:
    #     pickle.dump(linear_topo_configs, f)

    # todo: fix +1 bug in RP1, PCO8?
