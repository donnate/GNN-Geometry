import sys
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import math
from simulation_utils.shapes import *


def build_regular_structure(width_basis, basis_type,
                            shapes, start=0, start_color=0, add_random_edges=0,
                            col_increment  = 20,
                            plot=False, savefig=True):
    ''' This function creates a basis (torus, string, or cycle) and attaches elements of
    the type in the list regularly along the basis.
    Possibility to add random edges afterwards
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)
    start            :      initial nb for the first node
    add_random_edges :      nb of edges to randomly add on the structure
    plot,savefig     :      plotting and saving parameters
    OUTPUT:
    --------------------------------------------------------------------------------------
    Basis            :       a nx graph with the particular shape
    colors           :       labels for each role
    '''
    
    ### Sample (with replacement) where to attach the new motives
    Basis = eval(basis_type)(start, start_color, width_basis)
    start += Basis.n_nodes
    ### Sample (with replacement) where to attach the new motives
    nb_shapes = len(shapes)
    K = math.floor(width_basis / nb_shapes)
    plugins = [k * K for k in range(nb_shapes)]
    nb_shape = 0
    color_topology = Basis.labels
    labels_shape = Basis.labels_shape
    col_start = width_basis
    start = width_basis

    for s in range(len(shapes)):
        shape = shapes[s]
        type_shape = shape[0]
        if s == 0:
            seen_shapes = {'basis': 0, type_shape: col_start}
        else:
            if type_shape not in seen_shapes.keys():
                col_start += col_increment
                seen_shapes.update({type_shape: col_start})
        col_start = seen_shapes[type_shape]
        args = [start, col_start]
        if len(shape) > 1:
            args += shape[1:]
        print([type_shape] + args)
        S = eval(type_shape)(*args)
        ### Attach the shape to the basis
        Basis.add_motif(S,[(start, plugins[s])])
        color_topology += S.labels
        labels_shape += S.labels_shape
        color_topology[start]-=1
        color_topology[plugins[s]] -= seen_shapes[type_shape]
        start += S.n_nodes

    Basis.update_labels(color_topology)
    Basis.update_labels_shape(labels_shape)
    if add_random_edges>0:
        ## add random edges between nodes:
        for p in range(add_random_edges):
            src,dest=np.random.choice(nx.number_of_nodes(Basis),2, replace=False)
            Basis.add_edges_from([(src,dest)])
    return(Basis)


def build_hub_and_spokes_structure(n_nodes_per_core, n_nodes_per_periphery, n_cores=4,
                                   p_er=0.8, m_ba=1, k_links=1):
    ''' This function creates a basis (torus, string, or cycle) and attaches elements of
    the type in the list regularly along the basis.
    Possibility to add random edges afterwards
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)
    start            :      initial nb for the first node
    add_random_edges :      nb of edges to randomly add on the structure
    plot,savefig     :      plotting and saving parameters
    OUTPUT:
    --------------------------------------------------------------------------------------
    Basis            :       a nx graph with the particular shape
    colors           :       labels for each role
    '''
    a = 0
    labels = []
    labels_shape = []
    for i in range(0, n_cores):
        GG = nx.erdos_renyi_graph(n_nodes_per_core, p_er)
        labels_shape = ['ER Core -' + str(n_nodes_per_core)] * n_nodes_per_core
        start = n_nodes_per_core
        for j in range(n_nodes_per_core):
            P = nx.barabasi_albert_graph(n_nodes_per_periphery, m_ba)
            mapping = {k: k + start for k in range(n_nodes_per_periphery)}
            P = nx.relabel_nodes(P, mapping)
            GG.add_nodes_from(P.nodes)
            GG.add_edges_from(P.edges)
            GG.add_edge(j,  start)
            labels_shape += ['BA Periphery -' + str(n_nodes_per_periphery)] * n_nodes_per_periphery
            start += n_nodes_per_periphery
        mapping = {k: k + a for k in range(nx.number_of_nodes(GG))}
        GG = nx.relabel_nodes(GG, mapping)
        labels += [i] * nx.number_of_nodes(GG)
        
        if i == 0:
            Basis = Shape()
            Basis.G = GG
        else:
            Basis.G.add_nodes_from(GG.nodes)
            Basis.G.add_edges_from(GG.edges)
            #### Connect it randomly
            for k in range(k_links):
                e = np.random.choice(range(a))
                ee = np.random.choice(range(a, a + n_nodes_per_core))
                Basis.G.add_edge(e, ee)
        a += nx.number_of_nodes(GG)
        Basis.update_labels(labels)
        Basis.update_labels_shape(labels_shape)
        Basis.n_nodes = nx.number_of_nodes(Basis.G)
        
    return(Basis)
