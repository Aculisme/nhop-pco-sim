import os
from dataclasses import replace, dataclass
from pprint import pprint as pp
from typing import Callable, Optional

import matplotlib.pyplot as plt

import networkx as nx
import simpy
import numpy as np  # todo: replace with cupy on linux?
from numpy.lib.stride_tricks import sliding_window_view

import multiprocessing as mp
import time

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
    topo_params: Optional[dict] = None  # param : value, passed to generator
    # Setting random_seed will fix all trials to use the exact same random seed. If argument not passed, then each trial
    # will use a different time-based random seed.
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.topo_params is None:
            self.topo_params = {'n': self.num_nodes}
        # if self.random_seed is not None:
        #     self.topo_params['seed'] = self.random_seed


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
    fire_update_x: list
    fire_update_y: list
    out_of_sync_broadcast_x: list
    out_of_sync_broadcast_y: list


@dataclass
class InitState:
    id: int
    # initial_time_offset: int
    # clock_drift_rate: int
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
            LOGGING_ON=trial_config.logging_on,
            neighbors=neighbors_dict[i],
            phase_diff_percentage_threshold=trial_config.phase_diff_percentage_threshold,
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
        fire_update_x=[],
        fire_update_y=[],
        out_of_sync_broadcast_x=[],
        out_of_sync_broadcast_y=[],
    )

    env = simpy.Environment()
    # Create Nodes
    nodes = [trial_config.pco_node(env, init_state, logging) for init_state in init_states]

    for i, node in enumerate(nodes):
        node.all_nodes = nodes  # [nodes[n] for n in topo[i]]
        # node.all_nodes = [nodes[n] for n in range(len(nodes)) if n != i]  # [nodes[n] for n in topo[i]]
        # node.pos = pos[i]

    env.run(until=trial_config.sim_time)

    num_broadcasts = len(logging.fire_x)

    x = np.linspace(0, trial_config.sim_time, int(trial_config.sim_time / trial_config.reception_loop_ticks))

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

    time_until_synchronization = trial_config.sim_time

    # Find time until synchronization
    """We consider the network synchronized when the maximum pair-wise phase difference between all nodes at a given tick 
    is less than 2% of the period length, and remains so for 100% of the subsequent 100 ticks"""
    v = sliding_window_view(max_differences, 1000)
    for i, window in enumerate(v):
        if (window < trial_config.sync_epsilon).sum() >= trial_config.sync_num_ticks * len(window):
            time_until_synchronization = i * trial_config.reception_loop_ticks
            break

    avg_mean_phase_diff_after_synchronization = np.round(np.mean(mean_differences[-100:]), 2)
    avg_max_phase_diff_after_synchronization = np.round(np.mean(max_differences[-100:]), 2)
    time_until_synchronization_human_readable = np.round(time_until_synchronization / 1000, 2)

    # Print out metrics for the network
    b = time.time()
    print('processing time', round(b - a, 3), 's', '|',
          'TTS:', time_until_synchronization_human_readable, 's', '|',
          '#Broadcasts:', num_broadcasts, '|',
          'Synched max phase diff:', avg_max_phase_diff_after_synchronization, '%')

    # Generate plots
    if not trial_config.testing:
        fig, ax = plt.subplots(4, sharex=True)

        # Node phase
        for i in range(trial_config.num_nodes):
            ax[0].plot(logging.node_phase_x[i], logging.node_phase_y[i], label='node ' + str(i),
                       linewidth=(3 if i == 0 else 2),
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
        ax[1].plot(logging.fire_update_x, logging.fire_update_y, 'o', color='green', label='node update fire',
                   markersize=5)
        ax[1].plot(logging.out_of_sync_broadcast_x, logging.out_of_sync_broadcast_y, 'o', color='purple',
                   label='node out of sync broadcast',
                   markersize=5)

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
            trial_config.pco_node.name + ' Node Simulation for a ' + topo.__name__ + ' topology with ' + str(
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

    # todo: calculate properties about the graph, like connectivity, average degree, etc.
    # todo: store adjacency matrix, etc. for later analysis in different file?
    return Results(
        num_nodes=trial_config.num_nodes,
        num_broadcasts=num_broadcasts,
        avg_mean_phase_diff_after_synchronization=avg_mean_phase_diff_after_synchronization,
        avg_max_phase_diff_after_synchronization=avg_max_phase_diff_after_synchronization,
        time_until_synchronization_human_readable=time_until_synchronization_human_readable,
        rng_seed=rng_seed,
        ms_prob=trial_config.ms_prob,
    )


def worker(i, config, q):
    print("running trial", i, "of", config.num_trials)
    m = run_trial(config)
    res_string = m.to_csv()
    print(res_string)
    plt.show()
    q.put(res_string)
    return res_string


def writer(q, config):
    """listens for messages on the q, writes to file. """

    # create file with header if it doesn't exist
    if not os.path.isfile(config.file_out):
        with open(config.file_out, 'w+') as f:
            f.write(RESULTS_CSV_HEADER + "\n")

    with open(config.file_out, 'a+') as f:
        while True:
            s = q.get()
            if s == 'kill':
                print("done writing")
                break
            f.write(s + '\n')
            f.flush()


def main(config):
    # todo: replace with newer executor interface?
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(7)  # mp.cpu_count() + 2

    # put writer to work first
    w = pool.apply_async(writer, (q, config))

    # fire off workers
    jobs = []
    for i in range(config.num_trials):
        job = pool.apply_async(worker, (i, config, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # now we are done, kill the writer
    q.put('kill')
    pool.close()
    pool.join()


randomizedPcoNode5_config = TrialConfig(
    testing=False,
    logging_on=False,
    show_topo=False,
    overall_mult=1000,
    reception_loop_ticks=100,
    default_period_length=100 * 1000,
    sim_time=4000 * 1000,
    ms_prob=0.0,  # 1.0,  # todo: make optional?
    m_to_px=90,
    distance_exponent=15,
    clock_drift_rate_offset_range=0,#50,  # 100,
    clock_drift_variability=0.0,#0.05,
    min_initial_time_offset=0,
    max_initial_time_offset=199,
    sync_epsilon=2,
    sync_num_ticks=1,
    # todo: make file path dynamic? or not?
    file_out='/Users/lucamehl/Downloads/nhop-pco-sim/randomized_pco.txt',
    num_trials=1,
    phase_diff_percentage_threshold=0.02,
    k=2,
    num_nodes=32,
    # num_nodes=5,
    # pco_node=RandomizedPCONode4,
    pco_node=RandomizedPCONode5,
    # pco_node=PCONode8,
    topo=nx.path_graph,
    # topo=nx.barbell_graph,  # todo: note: num_nodes = m1*2 + m2
    # topo=nx.complete_graph,
    # topo=nx.random_internet_as_graph,
    # random_seed=1716582139984242000,
    # random_seed=1716631278258634000,
    random_seed=1716676895879946000,
    # topo_params={'m1': 5, 'm2': 1},  # seed
)

if __name__ == '__main__':
    # todo: make separate file for each or not? give them a name property?
    main(randomizedPcoNode5_config)
    # for i in range(0, 64):
    #     main(replace(randomizedPcoNode4_config, num_nodes=i, topo_params={'n': i}))
    # main(replace(randomizedPcoNode4_config, pco_node=RandomizedPCONode1))
    # main(replace(randomizedPcoNode4_config, pco_node=PCONode8, ms_prob=0))
    # main(replace(randomizedPcoNode4_config, pco_node=PCONode8, ms_prob=0))
