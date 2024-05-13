import sys

sys.path.append('lib')

import random
import networkx as nx
from nxsim import BaseNetworkAgent
from nxsim import BaseLoggingAgent
from nxsim import NetworkSimulation
import simpy
from matplotlib import pyplot as plt

# from transitions import Machine
from math import dist

'''

'''


class PCOAgent1(BaseNetworkAgent):
    def __init__(self, environment=None, agent_id=0, state=()):
        super().__init__(environment=environment, agent_id=agent_id, state=state)

        # initialise 1000 s timer
        # self.timer = self.env.timeout(1000, value='long')

        self.mainLoop = None
        self.period = 1000
        self.periodTimer = None
        self.neighbors = None

    # def neighbors_in_range(dist):
    #     pass

    # called (once) automatically when the agent is created
    def run(self):
        self.mainLoop = self.env.process(self.main())
        # run = self.env.process(self.run())
        # timeout = self.env.timeout(5)
        #
        # yield run | timeout
        # if not run.triggered:
        #     run.interrupt('message received')
        # pass
        # while True:
        # while self.timeCounter < self.period:

    def main(self):
        not_done = True
        self.periodTimer = self.env.timeout(self.period)
        while not_done:
            # long timer wakeup
            try:
                yield self.periodTimer
                not_done = False
            # message received interrupt
            except simpy.Interrupt as interrupt:
                print(interrupt.cause)
                if interrupt.cause == 'message received':
                    print('message received')
                else:
                    print('timer expired')

    # externally triggered callback for receiving a message
    def rx(self):
        self.mainLoop.interrupt('message received')
        pass





    def _tx(self):
        # normal_neighbors = self.get_neighboring_agents(state_id=0)
        normal_neighbors = self.get_neighboring_nodes()
        for neigh in normal_neighbors:
            neighbor = self.get_agent(neigh)
            if random.random() < neighbor.state[
                'rx_prob']:  # and neighbor.state['channel'][0] == self.state['channel'][0]
                neighbor.state['rx'].append(self.state['id'])
            # print("TimeStamp: ", self.env.now, self.id, neighbor.id, sep='\t')


def create_agent_network():
    THRESHOLD = 3  # Max TX range
    pos = [[0.1, 0.1, 3], [2, 0.1, 3],
           [0.1, 2, 2], [2, 2, 2],
           [0.1, 4, 2.5], [2, 4, 3]]
    # pos = pd.DataFrame(pos, columns=['x', 'y', 'z'])

    pos_tup = [(p[0], p[1], p[2]) for p in pos]
    pos_d = {r: pos_tup[r] for r in range(0, len(pos_tup))}
    edges_l = []

    for source in pos_d.items():
        ks = source[0]
        vs = source[1]
        for dest in pos_d.items():
            kd = dest[0]
            vd = dest[1]
            if ks == kd:
                continue
            else:
                if dist(vs, vd) < THRESHOLD and not (kd, ks) in edges_l:
                    # edges_l.append((ks,kd,dist(vs,vd))) # last value is distance as edge weight
                    edges_l.append((vs, vd, dist(vs, vd)))  # last value is distance as edge weight

    G = nx.Graph()
    G.add_weighted_edges_from(edges_l)

    # my_pos = nx.spring_layout(G, seed=3068)  # Seed layout for reproducibility
    nx.draw(G, with_labels=True)  # , pos=my_pos,
    plt.show()

    return G


# "id" refers to the current node state
# 'rx' is the message buffer of the node. The transmitting neighbor fills a nodes RX buffer.
# A jamming neighbour empties a node's RX buffer.
# A node acts on the values in its RX buffer every round and then
# clears the buffer at the end of the round.

if __name__ == '__main__':
    number_of_nodes = 6
    netw = create_agent_network()

    # Initialize agent states. Let's assume everyone is normal.
    init_states = [{'id': 0, 'rx': [], 'cancel': 0, 'rx_prob': 1.00} for _ in
                   range(number_of_nodes)]

    # Seed a new2 value to propagate
    init_states[5] = {'id': 1, 'rx': [], 'cancel': 0, 'rx_prob': 1.00}
    sim = NetworkSimulation(topology=netw, states=init_states, agent_type=FSM6,
                            max_time=40, dir_path='sim_01', num_trials=1, logging_interval=1.0)

    sim.run_simulation()

'''
node state machine:
    state = {
       'id': 0,  # 0 is normal, 1 is new2
       'rx': [],  # RX buffer
       'cancel': 0,  # Cancel flag
       'rx_prob': 1.00  # RX probability
    }
'''
