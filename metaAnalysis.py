import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# file_out = '/Users/lucamehl/Downloads/nhop-pco-sim/out/test2.txt'

def analyze_results(test_config, topo_configs, algo_configs):
    df = pd.read_csv(test_config.file_out)

    # print(df.columns)

    fig, ax = plt.subplots(2, sharex=True)

    colors = ['red', 'blue', 'purple', 'orange', 'pink', 'brown', 'grey']

    # for topo_config in topo_configs:
    for i, algo_config in enumerate(algo_configs):
        rows = df[(df['topo_name'] == 'linear') & (df['algo_name'] == algo_config.algo_name)]
        relevant = rows[['num_nodes', 'time_until_synchronization_human_readable', 'num_broadcasts']]
        gb = relevant.groupby('num_nodes', as_index=False)
        m = gb.mean()
        stdev = gb.std()
        # ax[0].plot(m['num_nodes'], m['time_until_synchronization_human_readable'], color=colors[i], label=algo_config.algo_name)
        # ax[1].plot(m['num_nodes'], m['num_broadcasts'], color=colors[i], label=algo_config.algo_name)

        # add errorbars
        ax[0].errorbar(m['num_nodes'], m['time_until_synchronization_human_readable'],
                       yerr=stdev['time_until_synchronization_human_readable'],
                       capsize=3, color=colors[i], fmt="--o", ecolor=colors[i], label=algo_config.algo_name)
        ax[1].errorbar(m['num_nodes'], m['num_broadcasts'], yerr=stdev['num_broadcasts'],
                       capsize=3, color=colors[i], fmt="--o", ecolor=colors[i], label=algo_config.algo_name)

    ax[0].set_ylabel('time_until_synchronization_human_readable')
    ax[1].set_ylabel('num_broadcasts')
    ax[1].set_xlabel('num_nodes')
    ax[0].legend()
    ax[1].legend()

    fig.suptitle(
        "Synchronization scaling over " + str(test_config.num_trials) + " trials"
    )

    plt.show()

    # time_until_synch = df['time_until_synchronization_human_readable']  # .values
    # print(time_until_synch)
    # time_until_synch_mean = np.mean(time_until_synch)
    # time_until_synch_st_dev = np.std(time_until_synch)
    # print("time_until_synch_mean", time_until_synch_mean)
    # print("time_until_synch_st_dev", time_until_synch_st_dev)
