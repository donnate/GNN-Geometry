import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Shape:
    def __init__(self, origin=0, origin_color=0, n_nodes=0, origin_community=0):
      self.origin = origin
      self.origin_color = origin_color
      self.n_nodes = n_nodes
      self.G = nx.Graph()
      self.G.add_nodes_from(range(origin, origin + n_nodes))
      self.labels = [origin_color] * n_nodes
      self.labels_shape = [np.nan] * n_nodes
    
    def add_motif(self, S, connection): 
        self.G.add_nodes_from(S.G.nodes())
        self.G.add_edges_from(S.G.edges())
        self.G.add_edges_from(connection)
        self.n_nodes += nx.number_of_nodes(S.G)
        self.labels += [S.labels]
        self.labels_shape += [S.labels_shape]
        
    def add_nodes(self, S, connection):
        self.G.add_nodes_from(S.G.nodes())
        self.G.add_edges_from(S.G.edges())
        self.G.add_edges_from(connection)
        self.n_nodes += nx.number_of_nodes(S.G)
        
    def update_labels(self, new_labels):
        self.labels = new_labels

    def update_labels_shape(self, new_labels):
        self.labels_shape = new_labels


    def draw(self, colour_by=None, cmap0=plt.cm.Set1):
        if colour_by == "topology":
            clmap = {v: k for k, v in enumerate(np.unique(self.labels))}
            nx.draw_networkx(self.G, pos=nx.layout.fruchterman_reingold_layout(self.G),
                             node_color=[clmap[u] for u in self.labels], cmap=cmap0)
        elif colour_by == "labels_shape":
            clmap = {v: k for k, v in enumerate(np.unique(self.labels_shape))}
            nx.draw_networkx(self.G, node_color=[clmap[u] for u in self.labels_shape], cmap=cmap0)
        else:
            nx.draw_networkx(self.G,cmap=cmap0)

    def saveplot(self, filename):
        plt.savefig(filename)

class TestGraph(Shape):
  def __init__(self):
      Shape.__init__(self, 0, 0, 30)

      self.labels = [0]
      self.labels += [1] * 7
      self.G.add_edges_from([(1,2), (1,3), (1,4), (2,4), (2,3), (4,5), (5,6), (6,7), (7,3), (7,4)])
      self.labels += [0] * 3
      self.G.add_edges_from([(11,12), (11,13), (11,14), (12,14), (12,13), (14,15), (15,16), (16,17), (17,13), (17,14)])
      self.labels += [2] * 7
      self.labels += [0] * 3
      self.G.add_edges_from([(21,22), (21,23), (21,24), (22,24), (22,23), (24,25), (25,26), (26,27), (27,23), (27,24)])
      self.labels += [3] * 7
      self.labels += [0] * 3
      self.G.add_edges_from([(7,8), (8,9), (9,10), (9,29), (8,30), (30,20), (20,8), (20,22), (22,18), (18,19), (18,12), (12,9), (18,8)])
      self.G.add_edges_from([(0,10), (29,0), (0,28)])
      self.G.add_edges_from([(19,31), (31,33), (19,32)])
      self.labels += [4] * 3


class Diamond(Shape):
    #### INPUT:
    #### -------------
    #### origin: "origining index" for the shape

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    def __init__(self, origin=0, origin_color=0, origin_community=0):
        Shape.__init__(self, origin, origin_color, 6, origin_community=origin_community)
        origin = self.origin
        self.G.add_edges_from([(origin, origin + 1), (origin + 1, origin + 2),
                      (origin + 2, origin + 3), (origin + 3, origin)])
        self.G.add_edges_from([(origin + 4, origin), (origin + 4, origin + 1),
                      (origin + 4, origin + 2), (origin + 4, origin + 3)])
        self.G.add_edges_from([(origin + 5, origin), (origin + 5, origin + 1),
                      (origin + 5, origin + 2), (origin + 5, origin + 3)])
        self.labels_shape = ['diamond'] * self.n_nodes
        self.labels = [ u + origin_color for u in self.labels]



class Cycle(Shape):
    #### INPUT:
    #### -------------
    #### origin: "origining index" for the shape

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    def __init__(self, origin=0, origin_color=0, n_nodes=10, origin_community=0):
        Shape.__init__(self, origin, origin_color, n_nodes=n_nodes,
                       origin_community=origin_community)
        self.shape_colour = [origin_color] * n_nodes
        self.community_colours = [0] * n_nodes
        self.labels = [0] * n_nodes

        for i in range(n_nodes - 1):
          self.G.add_edges_from([(origin + i, origin + i + 1)])
        self.G.add_edges_from([(origin + n_nodes - 1, origin)])
        self.labels_shape = ['cycle_' + str(n_nodes)] * self.n_nodes
        self.labels = [ u + origin_color for u in self.labels]

class House(Shape):
    #### INPUT:
    #### -------------
    #### origin: "origining index" for the shape

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    def __init__(self, origin=0, origin_color=0, origin_community=0):
        Shape.__init__(self, origin, origin_color, n_nodes=5,
                       origin_community=origin_community)
        self.G.add_nodes_from(range(origin, origin + 5))
        self.G.add_edges_from([(origin, origin + 1), (origin + 1, origin + 2), (origin + 2, origin + 3), (origin + 3, origin)])
        self.G.add_edges_from([(origin, origin + 2), (origin + 1, origin + 3)])
        self.G.add_edges_from([(origin + 4, origin), (origin + 4, origin + 1)])

        self.shape_colour = [origin_color] * self.n_nodes
        self.community_colours = [0] * self.n_nodes
        self.labels = [0, 0] + [1, 1]  + [2]
        self.labels = [ u + origin_color for u in self.labels]
        self.labels_shape = ['house'] * self.n_nodes
    


class Chain(Shape):
    #### INPUT:
    #### -------------
    #### origin: "origining index" for the shape

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    def __init__(self, origin=0, origin_color=0, n_nodes=2, origin_community=0):
        Shape.__init__(self, origin, origin_color, n_nodes=n_nodes,
                     origin_community=origin_community)
        for i in range(n_nodes-1):
            self.G.add_edges_from([(origin + i, origin + i+1)])
        self.shape_colour = [origin_color] * self.n_nodes
        self.community_colours = [0] * self.n_nodes
        self.labels = list(np.arange(n_nodes//2))
        if (n_nodes % 2) == 1:
            self.labels += [n_nodes//2]
        self.labels += list(np.arange(n_nodes//2))[::-1]
        self.labels = [ u + origin_color for u in self.labels]
        self.labels_shape = ['chain_'+ str(n_nodes)] * self.n_nodes
        

class Star(Shape):
    def __init__(self, origin=0, origin_color=0, n_nodes=5, origin_community=0):
        Shape.__init__(self, origin, origin_color, n_nodes=n_nodes,
                       origin_community=origin_community)
        
        for k in range(1,n_nodes):
          self.G.add_edges_from([(origin, origin + k)])

        self.labels = [0]  + [1] * (n_nodes - 1)
        self.labels_shape = ['star_'+ str(n_nodes)] * self.n_nodes


class Fan(Star):
    def __init__(self, origin=0, origin_color=0, n_nodes=5, origin_community=0):
        Star.__init__(self, origin, origin_color, n_nodes=n_nodes,
                      origin_community=origin_community)
        for k in range(1, self.n_nodes - 1):
          #self.labels[k - 1] += (k > 1) * (k <= n_nodes)
          self.labels[k] += 1
          self.labels[k + 1] += 1
          #if (k < self.n_nodes - 3): self.labels[k + 2] += 1
          self.G.add_edges_from([(origin + k, origin + k + 1)])
        self.labels[self.n_nodes - 2] += 1
        self.labels[2] += 1
        self.labels = [ u + origin_color for u in self.labels]
        self.labels_shape = ['fan_'+ str(n_nodes)] * self.n_nodes


class Clique(Shape):
    def __init__(self, origin=0, origin_color=0, n_nodes=5, origin_community=0,
                 nb_to_remove=0):
        Shape.__init__(self, origin, origin_color, n_nodes=n_nodes,
                       origin_community=origin_community)
        ### Defines a clique (complete graph on nb_nodes nodes, with nb_to_remove  edges that will have to be removed)
        A=np.ones((n_nodes, n_nodes))
        np.fill_diagonal(A, 0)
        self.G = nx.from_numpy_matrix(A)
        if nb_to_remove>0:
          edge_list = list(self.G.edges())
          lst=np.random.choice(len(edge_list), nb_to_remove, replace=False)
          to_delete=[edge_list[e] for e in lst]
          self.G.remove_edges_from(to_delete)
          for e in lst:
            self.labels[edge_list[e][0]] += 1
            self.labels[edge_list[e][1]] += 1
        mapping = {k: (k + origin) for k in range(n_nodes)}
        self.G = nx.relabel_nodes(self.G,mapping)
        self.labels_shape = ['clique_'+ str(n_nodes)] * self.n_nodes
        self.labels = [ u + origin_color for u in self.labels]


class Tree(Shape):
    #### INPUT:
    #### -------------
    #### origin: "origining index" for the shape
    #### n_levels: nb of levels in the tree
    #### n_children:  nb of children for each node
    def __init__(self, origin=0, origin_color=0, n_levels=2, n_children=2,
                 origin_community=0,
                 ):
        Shape.__init__(self, origin, origin_color, n_nodes= n_children**n_levels - 1,
                       origin_community=origin_community)
        self.n_levels = n_levels
        self.n_children = n_children
        self.labels = []

        nodes_level= np.concatenate([[l] * n_children**l for l in range(n_levels) ])
        a = origin
        b = 1 + origin
        it = 0
        for l in range(0, n_levels):
              #### Where the children start
            self.labels += [l] * (n_children**l)
           
            for nn in range(0, n_children**l):
                self.G.add_edges_from([(a + nn, b + nnn) for nnn in range(0, n_children)])
                b += n_children
            a += n_children**l
        self.labels += [n_levels] * (n_children**n_levels)
        self.labels = [ u + origin_color for u in self.labels]
        self.labels_shape = ['tree_'+ str(n_levels) + '_' + str(n_children)] * self.n_nodes


class Hollow(Shape):
    ### Creates a torus-like basis structure
        #### INPUT:
    #### -------------

    #### OUTPUT:
    #### -------------
    #### shape: a nx graph with the particular shape:
    def __init__(self, origin=0, origin_color=0,
                 origin_community=0
                 ):
        Cycle.__init__(self, origin, origin_color, n_nodes= 5,
                       origin_community=origin_community)
        G1 = Cycle(origin + 5, origin_color, n_nodes=10)
        self.G.add_nodes_from(G1.G.nodes())
        self.G.add_edges_from(G1.G.edges())
        self.G.add_edges_from([(origin, origin + 5), (origin + 1, origin + 7),
                               (origin + 2, origin + 9), (origin + 3, origin + 11), (origin + 4, origin + 13)])
        self.G.add_edges_from([(origin + 6, origin + 1), (origin + 6, origin)])
        self.G.add_edges_from([(origin + 8, origin + 2), (origin + 8, origin + 1)])
        self.G.add_edges_from([(origin + 10, origin + 3), (origin + 10, origin + 2)])
        self.G.add_edges_from([(origin + 12, origin + 4), (origin + 12, origin + 3)])
        self.G.add_edges_from([(origin + 14, origin), (origin + 14, origin + 4)])
        self.labels_shape = ['hollow'] * self.n_nodes
        self.labels = [ u + origin_color for u in self.labels]
        
        
class Neighbourhood(Shape):
  def __init__(self, origin, origin_color,
                n_neighbours = 3, radius=1,
                origin_community=0
                ):
    ####
    Shape.__init__(self, origin, origin_color, n_nodes = 0,
                       origin_community=origin_community)
   
    assignment = [1] + list([max([1, np.random.poisson(n_neighbours * u)]) for u in range(1, radius + 1)])
    assignment = list(np.sort(np.array(assignment)))
    self.n_nodes = np.sum(np.array(assignment))
    self.labels = list(np.concatenate([[i] * assignment[i] for i in range(0, radius + 1)]))
    self.G.add_edges_from([(origin,  origin + uu) for uu in range(1, 1+ assignment[1])])
    #### Randomly add edges between them too
    ####
    uu = np.random.choice(range(1, assignment[1] + 1), size=n_neighbours, replace=True)
    vv = np.random.choice(range(1, assignment[1] + 1), size=n_neighbours, replace=True)
    self.G.add_edges_from([(uu[i] + origin, origin +  vv[i]) for i in range(n_neighbours) if uu[i]!=vv[i]])

    #### Randomly add edges between other radii:
    if (radius>1):
      for i in range(2, 1 + radius):
        #### Ensure Connectedness
        uu = np.random.choice(list(np.where(np.array(self.labels) == i - 1)[0]),
                              size= np.sum(np.array(self.labels) == i), replace=True)
        self.G.add_edges_from([(uu[ii] +  origin, e + origin )
        for ii, e in enumerate(list(np.where(np.array(self.labels) == i)[0]))])

        uu = np.random.choice(list(np.where(np.array(self.labels) == i)[0]) +  list(np.where(np.array(self.labels) == i - 1)[0]),
                              size= assignment[i] * (n_neighbours-1), replace=True)
        vv = np.random.choice(list(np.where(np.array(self.labels) == i)[0]) +  list(np.where(np.array(self.labels) == i - 1)[0]),
                              size= assignment[i] * (n_neighbours-1), replace=True)
        try:
            self.G.add_edges_from([(uu[ii] + origin, origin +  vv[ii]) for ii in range(n_neighbours) if uu[ii]!=vv[ii]])
        except:
            pass

