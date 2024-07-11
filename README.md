# nhop-pco-sim
Multi-hop Pulse-Coupled Oscillators Simulation

## Repo information

`randomizedPcoNode5.py` contains the latest version of SyncWave that was tested in the simulation. It does _not_ contain the latest version of SyncWave. Only a pre-cursor.  

`pcoNode8.py` contains an implementation of a FiGo-like PCO algorithm, with epochs.

`randomizedPcoNode1.py` contains an implementation of the Randomized Phase paper (from Schmidt et al.) (with 50% phase response)

## Running the simulation

To configure the simulation, modify the `supervisor3.py` file to import the correct node files, then either:

- if you want to run a single trial with given parameters
    > Create a `TrialConfig` dataclass
    and call `run_trial` with that configuration (see example in `supervisor3.py`)
- or, to run a series of tests with different topologies, algorithms, and multiple trials for each
    > Create a list of `TopoConfigs`, a list of `AlgoConfigs`, and a `TestConfig` dataclass.
    Then, call `run_test` with these configurations. These results can then be analysed by calling `analyze_results` (modify as necessary). (see example in `supervisor3.py`)

Run the simulation with:

```bash
python3 supervisor3.py
```


[//]: # (## Aims)

[//]: # (- create a simulation tracking the number of epochs required for synchronization)

[//]: # (- and once synchronized, the number of epochs required to stay synchronized)

[//]: # (- the number of messages required )

[//]: # (- how long each epoch takes)

[//]: # (- different network configurations)

[//]: # (- different failure scenarios)

[//]: # (- different dynamic network configurations)

[//]: # (- comparing standard leader-based synchronization)

[//]: # (- with PCO &#40;multi-hop&#41;)

[//]: # (- and PCO + message suppression)

[//]: # (- and PCO + MS + SPR/LPR)

