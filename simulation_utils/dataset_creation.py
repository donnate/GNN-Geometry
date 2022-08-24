import torch
import numpy as np, networkx as nx
import math
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, from_networkx
import csrgraph as cg

def create_four_class_dataset(dataset_size, effect_size):
    edge_index = torch.empty([2, 24* dataset_size])
    X = np.zeros([13* dataset_size,1])
    Y = np.zeros([dataset_size])
    C = np.zeros([dataset_size])
    graph_list = [None]*dataset_size
    for i in np.arange(dataset_size):
        ##### Create all the stars
        if i< dataset_size//2:
            G = nx.balanced_tree(12,1)
        else:
            G = nx.balanced_tree(3,2)
        if i%2 ==0:
            loc = 0.0
        else:
            loc= effect_size  * np.sqrt(13)
        u = from_networkx(G)
        edge_index[:, (24*i): 24*(i+1)] =  u.edge_index + i * 13
        Xnp = np.random.normal(loc=loc, scale=1.0, size=13)
        X[13*i:13*(i+1),0] = Xnp
        Y[i] = np.mean(Xnp)
        C[i] = (i%2) + 2* (i//(dataset_size//2))
        graph_list[i] = Data(x=torch.from_numpy(X[13*i:13*(i+1),0].reshape([-1,1])).float(), y= C[i],edge_index=u.edge_index, num_nodes=13)

    Y= torch.from_numpy(Y).float()
    C= torch.from_numpy(C).int()
    X= torch.from_numpy(X.reshape([-1,1])).float()
    return X, Y, C, graph_list, edge_index

def create_four_class_dataset_with_paths(dataset_size, effect_size, nb_paths, length_path):
    edge_index = torch.empty([2, 24* dataset_size])
    X = np.zeros([nb_paths* dataset_size,length_path])
    Y = np.zeros([dataset_size])
    C = np.zeros([dataset_size])
    graph_list = [None]*dataset_size
    train_index = np.random.choice(np.arange(dataset_size), dataset_size//2)
    test_index = np.setdiff1d(np.arange(dataset_size), train_index)
    for i in np.arange(dataset_size):
        ##### Create all the stars
        if i< dataset_size//2:
            G = nx.balanced_tree(12,1)
        else:
            G = nx.balanced_tree(3,2)
        if i%2 ==0:
            loc = 0.0
        else:
            loc= effect_size / np.sqrt(13)
        u = from_networkx(G)
        G2 = nx.complete_graph(nb_paths)
        u2 = from_networkx(G2)
        edge_index[:, (24*i): 24*(i+1)] =  u.edge_index + i * 13
        Xnp = np.random.normal(loc=loc, scale=1.0, size=13)
        G2 = cg.csrgraph(G, threads=12) 
        walks = G2.random_walks(walklen=length_path, # length of the walks
                epochs=nb_paths, # how many times to start a walk from each node
                start_nodes=[0], # the starting node. It is either a list (e.g., [2,3]) or None. If None it does it on all nodes and returns epochs*G.number_of_nodes() walks
                return_weight=1.,
                neighbor_weight=1.
                )
        X[nb_paths*i:nb_paths*(i+1),:] = Xnp[walks]
        Y[i] = np.mean(Xnp)
        C[i] = (i%2) + 2* (i//(dataset_size//2))
        graph_list[i] = Data(x=torch.from_numpy(Xnp[walks]).float(), y= C[i], edge_index= u2.edge_index, num_nodes=nb_paths)

    Y= torch.from_numpy(Y).float()
    C= torch.from_numpy(C).int()
    X= torch.from_numpy(X.reshape([-1,1])).float()
    return X, Y, C, graph_list, edge_index


def create_three_class_interaction_dataset(dataset_size, graph_size):
    X = np.zeros([graph_size * dataset_size, 5])
    Y = np.zeros([dataset_size])
    C = np.zeros([dataset_size])
    graph_list = [None]*dataset_size
    for i in np.arange(dataset_size):
        ##### Start by creating the line
        G_core = (nx.balanced_tree(1,2))
        if i< dataset_size//3:
            XX = np.array([[1,0,0,0,0], [0,1,0,0, 0 ], [0,0,1,0,0]] )
            C[i] = 0
        elif i< 2*dataset_size//3:
            XX = np.array([[1,0,0,0,0], [0,0,1,0, 0 ], [0,1,0,0,0]] )
            C[i] = 1
        else:
            XX = np.array([[1,0,0,0,0], [0,0,1,0,0], [0,0,1,0,0]] )
            C[i] = 2
        G_core.add_edges_from([list([i, np.random.choice(np.arange(i))]) for i in np.arange(3,graph_size)])
        ##### Create random features vects:
        XXX = np.vstack([list((np.random.multinomial(1, [0,0,0,0.5, 0.5], size=1)).flatten()) 
                         for i in range(0,graph_size-3)])
        #### Create the rest of the graph
        u = from_networkx(G_core)
        Xnp = np.vstack([XX, XXX])
        X[(graph_size * i):graph_size * (i+1), :] = Xnp
        Y[i] = np.mean(Xnp)
        
        graph_list[i] = Data(x=torch.from_numpy(X[11*i:11*(i+1),:]).float(), y= C[i],
                             edge_index=u.edge_index, num_nodes=graph_size)

    Y= torch.from_numpy(Y).float()
    C= torch.from_numpy(C).int()
    X= torch.from_numpy(X.reshape([-1,1])).float()
    return X, Y, C, graph_list
