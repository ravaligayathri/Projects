# Author : Ravali Kuppachi

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
#matplotlib inline

# create state space and initial state probabilities

states = ['a', 'c', 'g' , 't']
pi = [0.25, 0.25 , 0.25 , 0.25]
state_space = pd.Series(pi, index=states, name='states')
#print(state_space)
#print(state_space.sum())
#print()

# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

q_df = pd.DataFrame(columns=states, index=states)
q_df.loc[states[0]] = [0.180, 0.274 , 0.426, 0.120]
q_df.loc[states[1]] = [0.171, 0.368, 0.274, 0.188]
q_df.loc[states[2]] = [0.161, 0.339, 0.375, 0.125]
q_df.loc[states[3]] = [0.079, 0.355, 0.384, 0.154]

#print(q_df)

q = q_df.values
#print('\n', q, q.shape, '\n')
#print(q_df.sum(axis=1))
#print()

from pprint import pprint

# create a function that maps transition probability dataframe
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(q_df)
#pprint(edges_wts)
#print()

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
states = ['a', 'c','g','t']
G.add_nodes_from(states)
#print(f'Nodes:\n{G.nodes()}\n')
#print('Nodes:\n', G.nodes(), "\n")

# edges represent transition probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
#print(f'Edges:')
#print("Edges:")
#pprint(G.edges(data=True))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)
# In Windows: dot -Tps filename.dot -o outfile.ps


# create edge labels for jupyter plot but is not necessary
edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'pet_dog_markov.dot')


# create state space and initial state probabilities

hidden_states = ['cpg', 'noncpg']
pi = [0.5, 0.5]
state_space = pd.Series(pi, index=hidden_states, name='states')
#print(state_space)
#print('\n', state_space.sum())
#print()

# Next we create our transition matrix for the hidden states.
# create hidden transition matrix
# a or alpha = transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]
a_df.loc[hidden_states[1]] = [0.4, 0.6]

#print(a_df)

a = a_df.values
#print('\n', a, a.shape, '\n')
#print(a_df.sum(axis=1))
#print()

# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given state
# matrix is size (M x O) where M is number of states
# and O is number of different possible observations

observable_states = ['a', 'c', 'g' , 't']
#print()
#print("observable_states:\n", states)
#print("hidden_states:\n", hidden_states)
#print()

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.155, 0.341, 0.350, 0.154]
b_df.loc[hidden_states[1]] = [0.262, 0.246, 0.239, 0.253]

#print(b_df)

b = b_df.values
#print('\n', b, b.shape, '\n')
#print(b_df.sum(axis=1))
#print()


# Now we create the graph edges and the graph object.
# create graph edges and weights

hide_edges_wts = _get_markov_edges(a_df)
#pprint(hide_edges_wts)

emit_edges_wts = _get_markov_edges(b_df)
#pprint(emit_edges_wts)
#print()

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(hidden_states)
#print(f'Nodes:\n{G.nodes()}\n')
#print('Nodes:\n', G.nodes(), '\n')

# edges represent hidden probabilities
for k, v in hide_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

# edges represent emission probabilities
for k, v in emit_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

#print('Edges:')
#pprint(G.edges(data=True))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='neato')
nx.draw_networkx(G, pos)

# create edge labels for jupyter plot but is not necessary
emit_edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=emit_edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'pet_dog_hidden_markov.dot')
# In Windows: dot -Tps filename.dot -o outfile.ps

#print()

def viterbi(pi, a, b, obs):

    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))

    # init delta and phi
    delta[:, 0] = pi * b[:, int(obs[0])]
    phi[:, 0] = 0

    #print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            #print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))

    # find optimal path
    #print('-'*50)
    #print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1])) #LPW
    for t in range(T-2, -1, -1):
        path[t] = phi[int(path[t+1]), [t+1]]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1])) #LPW
        #print('path[{}] = {}'.format(t, path[t]))

    return path, delta, phi

# observation sequence of dog's behaviors
# observations are encoded numerically
obsarray=[]
with open("dna_seq.txt") as infile:
    for line in infile:
        obslist=line.replace('a','0,').replace('c','1,').replace('g','2,').replace('t','3,').splitlines()

        obsarray.append(np.array(ast.literal_eval(obslist[0])))#str1=''.join(line1)
    #print(obsarray)

    for obs in obsarray:
        obs_map = {'a':0, 'c':1 , 'g':2 , 't':3}
        inv_obs_map = dict((v,k) for k, v in obs_map.items())
        obs_seq = [inv_obs_map[v] for v in list(obs)]

        #print( pd.DataFrame(np.column_stack([obs, obs_seq]),
        #columns=['Obs_code', 'Obs_seq']) )
        #print(obs)
        path, delta, phi = viterbi(pi, a, b, obs)
        #print('\nsingle best state path: \n', path)
        #print('delta:\n', delta)
        #print('phi:\n', phi)
        #print()
        state_map = {0:'I', 1:'N'}
        state_path = [state_map[v] for v in path]

        #print()
        #print("RESULT:")
        print(line)
        str1 = ''.join(state_path)
        print(str1)
        #print(np.array(ast.literal_eval(state_path[0])))
