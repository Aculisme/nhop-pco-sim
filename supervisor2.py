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

from pcoNode8 import reception_probability, distance_euc, InitState, Logging, PCONode8, Results, \
    RESULTS_CSV_HEADER
from randomizedPcoNode1 import RandomizedPCONode1
from randomizedPcoNode2 import RandomizedPCONode2


@dataclass
class TrialConfig:
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
    sync_epsilon: int
    sync_num_ticks: float
    file_out: str
    num_trials: int
    pco_node: Callable
    topo: Callable  # nx.Graph generator function
    topo_params: Optional[dict] = None  # param name : value, passed to generator
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.topo_params is None:
            self.topo_params = {'n': self.num_nodes}
        if self.random_seed is not None:
            self.topo_params['seed'] = self.random_seed


def run_trial(trial_config):
    a = time.time()

    # Generate graph plot and positions
    topo = trial_config.topo
    G = topo(**trial_config.topo_params)

    pos = nx.nx_agraph.graphviz_layout(G)

    rng_seed = time.time_ns()
    if trial_config.random_seed is not None:
        rng_seed = trial_config.random_seed

    reception_probabilities = {i: {} for i in range(trial_config.num_nodes)}
    for x, i in enumerate(pos):
        for y, j in enumerate(pos):
            if i != j:
                reception_prob = reception_probability(distance_euc(pos[i], pos[j]) / trial_config.m_to_px,
                                                       trial_config.distance_exponent)
                reception_probabilities[x][y] = round(reception_prob, 3)

    if not trial_config.testing:
        print('reception_probabilities:')
        pp(reception_probabilities)

    if not trial_config.testing:
        nx.draw(G, with_labels=True, pos=pos)

    # initial node states
    init_states = [
        InitState(
            id=i,
            clock_drift_scale=trial_config.clock_drift_variability,
            min_initial_time_offset=trial_config.min_initial_time_offset,
            max_initial_time_offset=trial_config.max_initial_time_offset,
            clock_drift_rate_offset_range=trial_config.clock_drift_rate_offset_range,
            OVERALL_MULT=trial_config.overall_mult,
            RECEPTION_LOOP_TICKS=trial_config.reception_loop_ticks,
            DEFAULT_PERIOD_LENGTH=trial_config.default_period_length,
            M_TO_PX=trial_config.m_to_px,
            DISTANCE_EXPONENT=trial_config.distance_exponent,
            MS_PROB=trial_config.ms_prob,
            pos=pos[i],
            rng_seed=rng_seed + i,
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
    nodes = [trial_config.pco_node(env, init_state, logging) for init_state in init_states]

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
    # print('Number of broadcasts:', num_broadcasts)
    # print("Synchronized avg. phase diff: ", avg_mean_phase_diff_after_synchronization)
    # print("Time until synchronization: ", time_until_synchronization_human_readable)
    b = time.time()
    # print('Processing took', round(b - a, 3), 'seconds')
    print('processing time', round(b - a, 3), 's', '|',
          'TTS:', time_until_synchronization_human_readable, 's', '|',
          '#Broadcasts:', num_broadcasts, '|',
          'Synched max phase diff:', avg_max_phase_diff_after_synchronization, '%')

    # Generate plots
    if not trial_config.testing:
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
            'Randomized PCO (Schmidt et al.) Node Simulation for a ' + topo.__name__ + ' topology with ' + str(
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
        ]

        for i, metric_text in enumerate(metrics):
            plt.text(.50, .96 - .1 * i, metric_text, ha='left', va='top', transform=ax[3].transAxes)

        plt.show()

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


default_config = TrialConfig(
    testing=True,
    logging_on=False,
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
    sync_epsilon=2,
    sync_num_ticks=1.0,
    # todo: make file path dynamic? or not?
    file_out='/Users/lucamehl/Downloads/nhop-pco-sim/temp.txt',
    num_trials=100,
    num_nodes=50,
    topo=nx.random_internet_as_graph,
    pco_node=PCONode8,

    # Setting random_seed will fix all trials to use the exact same random seed. If argument not passed, then each trial
    # will use a different time-based random seed.
    # random_seed=0,
    # topo_params={'n': 10, 'seed': 0}
)

randomized_pco_config = TrialConfig(
    testing=False,  #
    logging_on=False,
    overall_mult=1000,
    reception_loop_ticks=100,
    default_period_length=100*100,#100 * 1000,
    sim_time=2000 * 1000,
    ms_prob=1.0,  # todo: make optional?
    m_to_px=90,
    distance_exponent=15,
    clock_drift_rate_offset_range=100,
    clock_drift_variability=0.05,
    min_initial_time_offset=0,
    max_initial_time_offset=200,
    sync_epsilon=2,
    sync_num_ticks=1.0,
    # todo: make file path dynamic? or not?
    file_out='/Users/lucamehl/Downloads/nhop-pco-sim/randomized_pco.txt',
    num_trials=1,  #
    num_nodes=11,  #
    pco_node=RandomizedPCONode2,
    topo=nx.barbell_graph,  # todo: note: num_nodes = m1*2 + m2
    # topo=nx.complete_graph,
    # Setting random_seed will fix all trials to use the exact same random seed. If argument not passed, then each trial
    # will use a different time-based random seed.
    # random_seed=0,
    topo_params={'m1': 5, 'm2': 1}  # seed
)


# thirty_node_config = replace(default_config, num_nodes=30)

# ninety_ms_config = replace(default_config, ms_prob=0.9)
# eighty_ms_config = replace(default_config, ms_prob=0.8)


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


if __name__ == '__main__':
    # todo: make separate file for each or not? give them a name property?
    print("testing randomized pco config")
    main(randomized_pco_config)
    # print("testing 90% ms prob config")
    # main(ninety_ms_config)
    # print("testing 80% ms prob config")
    # main(eighty_ms_config)
