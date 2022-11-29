from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import pandas as pd
import  seaborn as sb
from utils.curvature import *
from utils.helper import *


def diffusion_dist(data, epsilon = 0.5, option = 'node'):
    deg1 = deg(data.edge_index)
    G = to_networkx(data)
    if option == "node":
        diff = sc.spatial.distance.cdist(data.x, data.x)
    else:
        short_dist = np.zeros((len(deg1),len(deg1)))
        # length = dict(nx.all_pairs_shortest_path_length(G))
        for i in range(len(deg1)):
            for j in range(i+1,len(deg1)):
                try:
                    short_dist[i,j] = nx.shortest_path_length(G,i,j)
                except nx.NetworkXNoPath:
                    short_dist[i,j] = 9999
        d = short_dist
        K = np.exp((-d**2)/0.5)
        deg_inv = torch.sum(Tensor(K),dim = 1)**(-1)
        deg_inv = deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        h = PCA(n_components = 5).fit(torch.mm(torch.diag(deg_inv), Tensor(K)))
        dd = PCA(n_components = 5).fit_transform(torch.mm(torch.diag(deg_inv), Tensor(K)))
        A = torch.mm(Tensor(dd), torch.diag(Tensor(h.singular_values_)))
        diff = sc.spatial.distance.cdist(A, A)
    return(diff)


def dist_corr(data, embedding, name='Cora', trial=1):
    alpha = np.arange(0,1.1,0.1)
    experiment = ["normal","diffusion"]
    d = {}
    n = data.num_nodes
    dist_f = diffusion_dist(data, option = 'node')
    dist_g = diffusion_dist(data, option = 'graph')
    for i in range(len(experiment)):
        for j in range(10):
                temp = embedding[name+'_'+experiment[i]+ '_' +str(alpha[j+1])+'_'+'loop'+'_'+str(True)+'_'+str(trial)]['embedding']
                dist = sc.spatial.distance.cdist(temp.detach().numpy(), temp.detach().numpy())
                d[name+'_'+experiment[i]+ '_' +str(alpha[j+1])+'_'+'loop'+'_'+str(True)+'_'+str(trial)] = {
                    'mean_dist' : dist.mean(),
                    'sd_dist' : dist.std(),
                    'corr_kt_g' : kendalltau(dist_g[np.triu_indices(n)],
                                                    dist[np.triu_indices(n)])[0],
                    'corr_sp_g' : spearmanr(dist_g[np.triu_indices(n)],
                                                   dist[np.triu_indices(n)])[0],
                    'corr_kt_f' : kendalltau(dist_f[np.triu_indices(n)],dist[np.triu_indices(n)])[0],
                    'corr_sp_f' : spearmanr(dist_f[np.triu_indices(n)],dist[np.triu_indices(n)])[0],
                    'gnn_type' : experiment[i],
                    'alpha' : alpha[j+1],
                    'add_self_loop' : True,
                    'dataset' : name
                                                             }
    return(d)


def draw_corr(result, name='Cora', option="graph"):
    df = pd.DataFrame.from_dict(result).T
    df.corr_sp_g = df.corr_sp_g.astype('float32')
    df.corr_sp_f = df.corr_sp_f.astype('float32')
    sb.set_theme(style="white", rc = {'figure.figsize':(9,6)})
    if option == "graph":
        G = sb.lineplot(data = df, x = 'alpha', y = 'corr_sp_g', hue = 'gnn_type', linewidth = 5)
    else:
        G = sb.lineplot(data = df, x = 'alpha', y = 'corr_sp_f', hue = 'gnn_type', linewidth = 5)
    G.set_xticks(np.arange(0,1.1,0.1))
    G.axhline(0, ls='--', c = 'black')
    G.set_xticklabels = (['0',
                '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])
    G.legend(labels = ['Symmetric','Row-normalized'],loc = 'upper left', fontsize =20)
    G.set_ylabel('Spearman Correlation', fontsize = 20)
    G.set_xlabel(r'$\alpha$', fontsize = 20)
    plt.show()


def curv_corr(data, embedding, name='Cora', trial=1, loop=True):
    F_embeddings = {}
    alpha = np.arange(0, 1.1, 0.1)
    experiment = ["normal", "diffusion"]
    Adj = nx.adjacency_matrix(to_networkx(data)).todense()
    F = augmented_forman(Adj)
    for i in range(2):
        for j in range(10):
            temp = embedding[
                name + '_' + experiment[i] + '_' + str(alpha[j + 1]) + '_' + 'loop' + '_' + str(loop) + '_' + str(
                    trial)]['embedding']
            F1 = augmented_forman(pre_curvature(temp, data)).detach().numpy()
            F_embeddings[
                name + '_' + experiment[i] + '_' + str(alpha[j + 1]) + '_' + 'loop' + '_' + str(loop) + '_' + str(
                    trial)] = {
                'curvature': F1,
                'corr_kt': kendalltau(F, F1)[0],
                'corr_sp': spearmanr(F, F1)[0],
                'corr_pr': pearsonr(F, F1)[0],
                'gnn_type': experiment[i],
                'alpha': alpha[j + 1],
                'add_self_loop': loop,
                'dataset': name
            }
    tab = pd.DataFrame.from_dict(F_embeddings).T
    tab.corr_sp = tab.corr_sc.astype('float')

    sb.set_theme(style="white", rc={'figure.figsize': (9, 6)})
    G = sb.lineplot(data=tab, x='alpha', y='corr_sp', hue='gnn_type', linewidth=5)
    G.set_xticks(np.arange(0, 1.1, 0.1))
    G.axhline(0, ls='--', c='black')
    G.set_xticklabels = (['0',
                          '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])
    G.legend(labels=['Symmetric', 'Row-normalized'], loc='upper left', fontsize=20)
    G.set_ylabel('Spearman Correlation', fontsize=20)
    G.set_xlabel(r'$\alpha$', fontsize=20)
    plt.show()
