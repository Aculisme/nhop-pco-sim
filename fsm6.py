import sys

sys.path.append('lib')

import random
import networkx as nx
from nxsim import BaseNetworkAgent
from nxsim import BaseLoggingAgent
from nxsim import NetworkSimulation
from matplotlib import pyplot as plt

# from transitions import Machine
from math import dist

'''
FSM6, March 10th 2023
Adding channels, and Jamming range.

A simple network that 'gossips' a value from node to node.
There is a probability that a node introduces a new value to propagate,
and a probability that a node becomes evil.
There is a probability that a node will receive or drop a transmission
from a neighbour.

An evil node 'jams' all of it neighbours by reducing their probability of message
reception

Location: The nodes now use their position, a 3d tuple of floats representing
(x,y,z), as their ID. 

Channels: The nodes have a list of channels. Normal, non-malicious, nodes will
only have one channel. Nodes choose a new channel randomly each time period.
I need to think deeper about the channel hopping scheme so that we only measure 
the impact of a jammer, and not of a stupid channel hopping scheme.

The malicious node may have multiple channels. This means that it may communicate 
or jam multiple channels simultaneously. 

Channel hopping scheme: could be random from the set of channels, but then there is a
chance that two neighbour nodes will never communicate. That chance increases with the
number of available channels. Round Robin could also fail if two nodes start at different
channels and their channels never become the same. The best option is a scheme that guarantees
that every node communicates with all neighbours once a round. Hmmmm.... But, then 
a jammer could jamm communication, or use jamming to disrupt the channel hopping scheme.

If a node visits each neighbor in round-robin fashion, then jamming a node could 
cause its neighbors to block waiting for communication from the attacked node. A 
liveness attack. Or, nodes may not block, and loose the guarantee of communication 
with each neighbor each time period, and weaken it to eventual communication with each 
neighbour. Hmmm...

Jamming Range: The jamming model is now unit disc, not graph based. The nodes IDs are 
the positions, and they are used to determine if a node is in the jamming radius.
NOTE: with this modification I could get rid of NetworkX, and only use SimPy. I leave
that for later.

Cost: Cost is yet to be done, but could be something as simple as number of channels 
times the range, summed up over time. That would capture the notion of a jammer that 
dynamically changes its jamming targets and schedule to avoid detection.

'''


class FSM6(BaseNetworkAgent):
    def __init__(self, environment=None, agent_id=0, state=()):
        super().__init__(environment=environment, agent_id=agent_id, state=state)
        # Can an node be jamming and jammed, YES!
        self.new_prob = 0.01
        self.evil_prob = 0.01
        self.neighbors_heard = {}

    def neighbors_in_range(dist):
        pass

    # state machine
    def run(self):
        pass
        # while True:
        #     pass
            # # Update local state and send message to update network
            # if random.random() < self.new_prob:  # increment local value
            #     self.state['id'] += 1
            #     print("Update:", self.env.now, self.id, "New Val:", self.state['id'], sep=' ')
            #
            # # Become evil, block all RX from neighbours close to you.
            # if random.random() < self.evil_prob:  # increment local value
            #     self.state['evil'] = 1
            #     self.state['channel'].append(1)
            #     self.state['channel'].append(2)
            #     print("EVIL:", self.env.now, self.id, 'channels:', self.state['channel'], sep=' ')
            #
            # # Test if a neighbour is evil and I am jammed.
            # self.jammed = self.is_jammed()
            #
            # # State machine for RX values
            # if self.state['rx'] and max(self.state['rx']) > self.state['id']:  # Update my state
            #     self.state['id'] = max(self.state['rx'])
            #     self.state['cancel'] = 0
            # elif self.state['rx'] and min(self.state['rx']) < self.state['id']:  # Send my state to neighbour
            #     self.state['cancel'] = 0
            # elif self.state['rx'] and len(self.get_agents(limit_neighbors=True)) == len(self.state['rx']) and (
            #         len(self.state['rx']) == len([x for x in self.state['rx'] if x == self.state['id']])):
            #     self.state['cancel'] = 1  # be polite and save energy
            # else:
            #     pass
            #
            # if not self.state['cancel'] == 1:
            #     self.tx()
            #
            # print("TS:", self.env.now, self.id, self.state['id'], self.state['channel'][0], "cancel:",
            #       self.state['cancel'], self.state['rx'], sep=' ')
            # if (self.id == 5):
            #     print()
            # self.state['cancel'] = 0
            # self.state['rx'].clear()
            # if not self.state['evil']:
            #     self.state['channel'][0] = random.choice([0, 1, 2])
            # # change timeout val to get different node activation ordering.
            # # yield self.env.timeout(1*random.random())
            # yield self.env.timeout(1)

    def tx(self):
        # normal_neighbors = self.get_neighboring_agents(state_id=0)
        normal_neighbors = self.get_neighboring_nodes()
        for neigh in normal_neighbors:
            neighbor = self.get_agent(neigh)
            if random.random() < neighbor.state['rx_prob']: #  and neighbor.state['channel'][0] == self.state['channel'][0]
                neighbor.state['rx'].append(self.state['id'])
            # print("TimeStamp: ", self.env.now, self.id, neighbor.id, sep='\t')

    # def is_jammed(self):
    #     # normal_neighbors = self.get_neighboring_agents(state_id=0)
    #     JAMM_RANGE = 5
    #     # normal_neighbors = self.get_neighboring_nodes()
    #     neighbors_in_range = [x for x in self.get_all_agents() if dist(x.id, self.id) < JAMM_RANGE]
    #     for neigh in neighbors_in_range:
    #         nbor = self.get_agent(neigh.id)
    #         if nbor.state['evil'] == 1 and nbor.state['channel'][0] in self.state['channel']:
    #             self.state['rx_prob'] = 0.90
    #             return 1
    #             # print("TimeStamp: ", self.env.now, self.id, neighbor.id, sep='\t')
    #     return 0


# def create_network():
#     THRESHOLD = 3
#     pos = [[0.1, 0.1, 3], [2, 0.1, 3],
#            [0.1, 2, 2], [2, 2, 2],
#            [0.1, 4, 2.5], [2, 4, 3]]
#     # pos = pd.DataFrame(pos, columns=['x', 'y', 'z'])
#     pos_d = {r: pos[r] for r in range(0, len(pos))}
#
#     edges_l = []
#
#     for ks, vs in pos_d.items():
#         for kd, vd in pos_d.items():
#             if ks == kd:
#                 continue
#             else:
#                 if dist(vs, vd) < THRESHOLD and not (kd, ks) in edges_l:
#                     edges_l.append((ks, kd, dist(vs, vd)))  # last value is distance as edge weight
#
#     G = nx.Graph()
#     G.add_weighted_edges_from(edges_l)
#
#     return G


# Fun fact: we model the valid network of non-malicious
# nodes as a graph, with links made between nodes within
# tansmission range.
# The Jamming node is a node in the graph, but jamming is
# done on euclidian distance from jamming node based on the
# 3d coordinates of the jammed nodes.
# This may allow me to factor out the use of the graph in the
# future, but is useful now.

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

    # nx.draw_networkx(G, pos_d)
    # # Set margins for the axes so that nodes aren't clipped
    # ax = plt.gca()
    # ax.margins(0.20)
    # plt.axis("off")
    # plt.show()

    # my_pos = nx.spring_layout(G, seed=3068)  # Seed layout for reproducibility
    nx.draw(G, with_labels=True) # , pos=my_pos,
    plt.show()

    return G


# "id" refers to the current node state
# 'rx' is the message buffer of the node. The transimitting neighbor fills a nodes RX buffer.
# A jamming neighbour empties a node's RX buffer.
# A node acts on the values in its RX buffer every round and then
# clears the buffer at the end of the round.

# CHANNELS: channels are a list. Normal nodes have only one channel.
# Malicious nodes have a list of channels and may jam on more than
# on channel at a time.

if __name__ == '__main__':
    number_of_nodes = 6
    netw = create_agent_network()



    # Initialize agent states. Let's assume everyone is normal.
    init_states = [{'id': 0, 'rx': [], 'cancel': 0, 'evil': 0, 'rx_prob': 1.00, 'channel': [0]} for _ in
                   range(number_of_nodes)]

    # Seed a new2 value to propagate
    init_states[5] = {'id': 1, 'rx': [], 'cancel': 0, 'evil': 0, 'rx_prob': 1.00, 'channel': [0]}
    sim = NetworkSimulation(topology=netw, states=init_states, agent_type=FSM6,
                            max_time=40, dir_path='sim_01', num_trials=1, logging_interval=1.0)

    sim.run_simulation()

