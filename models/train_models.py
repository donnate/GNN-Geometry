import numpy as np
from carbontracker.tracker import CarbonTracker
import cProfile
import os
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix
import time
from models.dgi import DGI
from models.mvgrl import MVGRL
from models.grace import GRACE
from models.baseline_models import GNN
#from models.cca_ssg import CCA_SSG
#from models.bgrl import BGRL
from models.data_augmentation import *
#from models.clgr import CLGR
from models.vgnae import *
from scipy import optimize
import scipy

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from codecarbon import OfflineEmissionsTracker
#from umap_functions import *


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.uniform_(m.weight, a=0.0, b=10.0)
        m.bias.data.fill_(0.1)



def train_dgi(data, hid_dim, out_dim, n_layers, patience=20,
              epochs=200, lr=1e-3, name_file="1", device=None,
              gnn_type="symmetric", alpha=0.5, beta = 1,add_self_loops= True, dropout_rate: float = 0.5):

    log_dir = '/log_dir/log_dir_DGI_' + str(out_dim)+ '/'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = data.num_features
    N = data.num_nodes
    cnt_wait = 0
    best = 1e9
    best_t = 0
    ##### Train DGI model #####
    print("=== train DGI model ===")
    model = DGI(in_dim, hid_dim, out_dim, n_layers, dropout_rate, gnn_type, alpha, beta, add_self_loops)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    loss_fn1 = nn.BCEWithLogitsLoss()
    #tracker = OfflineEmissionsTracker(country_iso_code="US", project_name='DGI_'+ str(out_dim) + '_' +  name_file)
    #tracker.start()
    for epoch in range(epochs):
        #tracker.epoch_start()
        tic_epoch = time.time()
        model.train()
        optimizer.zero_grad()
        idx = np.random.permutation(N)
        shuf_fts = data.x[idx,:]
        lbl_1 = torch.ones(1, N)
        lbl_2 = torch.zeros(1, N)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(data, shuf_fts)
        loss = loss_fn1(logits, lbl)

        loss.backward()
        optimizer.step()
        #tracker.epoch_end()

        print('Epoch={:03d}, loss={:.4f}, time={:.4f}'.format(epoch, loss.item(), time.time()-tic_epoch))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), os.getcwd()  +
                                           '/results/best_dgi_dim' + str(out_dim) + '_' + name_file +  '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break
    #tracker.stop()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(os.getcwd()   + '/results/best_dgi_dim'
                                     + str(out_dim) + '_' + name_file +  '.pkl'))
    return(model)


def train_mvgrl(data, diff, out_dim, n_layers, patience=20,
              epochs=200, lr=1e-3, wd=1e-4, name_file="1",
              device=None, gnn_type="symmetric", alpha=0.5, beta = 1,add_self_loops= True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_dim = data.num_features
    N = data.num_nodes
    cnt_wait = 0
    best = 1e9
    best_t = 0

    lbl_1 = torch.ones(N * 2) #sample_size
    lbl_2 = torch.zeros(N * 2) #sample_size
    lbl = torch.cat((lbl_1, lbl_2))

    model = MVGRL(in_dim, out_dim, gnn_type, alpha, beta, add_self_loops) # hid_dim, , n_layers
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn1 = nn.BCEWithLogitsLoss()
    #tracker = OfflineEmissionsTracker(country_iso_code="US", project_name='MVGRL_'+ str(out_dim) + '_' +  name_file)
    #tracker.start()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # sample_idx = torch.LongTensor(random.sample(node_list, N)) # sample_size
        # sample = data.subgraph(sample_idx)
        # Dsample = diff.subgraph(sample_idx)
        shuf_idx = np.random.permutation(N) #sample_size
        shuf_fts = data.x[shuf_idx,:]
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(data, diff, shuf_fts)  # sample, Dsample
        loss = loss_fn1(logits, lbl)
        loss.backward()
        optimizer.step()
        # pdb.set_trace()
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), os.getcwd()  +
                       '/results/best_mvgrl_dim' + str(out_dim) + '_' + name_file +  '.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break
    #tracker.stop()
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(os.getcwd()  + '/results/best_mvgrl_dim' +
                                     str(out_dim) + '_' + name_file +  '.pkl'))
    return(model)



def train_gnumap(data, hid_dim, dim, n_layers=2, target=None,
                 method = 'laplacian', must_propagate=None,
                 norm='normalize', neighbours=15,
                 patience=20, epochs=200, lr=1e-3, wd=1e-2,
                 min_dist=0.1, name_file="1", subsampling=None,
                 alpha: float=0.5, spread = 1.0, lambd_corr=1e-2,
                 beta: float=1., gnn_type: str = 'symmetric',
                 repulsion_strength=None,
                 local_connectivity=1,
                 device=None, colours=None):


    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPS_0 = data.num_edges/ (data.num_nodes ** 2)
    _a, _b = find_ab_params(spread, min_dist)

    if torch_geometric.utils.is_undirected(data.edge_index):
        new_edge_index, new_edge_attr = torch_geometric.utils.to_undirected(data.edge_index, data.edge_weight)
    else:
        new_edge_index, new_edge_attr = data.edge_index, data.edge_weight
    ### remove self loop
    #### transform edge index into knn matrix
    knn = []
    for i in range(data.num_nodes):
        knn += [list(np.sort(list(new_edge_attr[(new_edge_index[0]==i) & (new_edge_index[1]!=i )].numpy())))]
    knn_dists = pd.DataFrame(knn).fillna(0).values
    sigmas, rhos = smooth_knn_dist(
            knn_dists,
            float(neighbours),
            local_connectivity=float(local_connectivity),
             )
    # vals = [ np.exp(-(np.max(new_edge_attr.numpy()[i] - rhos[new_edge_index[0,i]], 0)) /
    #                 (sigmas[new_edge_index[0,i]])) for i in range(len(new_edge_attr))]
    vals = new_edge_attr
    rows = new_edge_index[0,:].numpy()
    cols = new_edge_index[1,:].numpy()
    vals = np.array(vals)
    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(data.x.shape[0], data.x.shape[0])
    )
    result.eliminate_zeros()
    target_graph_index, target_graph_weights = from_scipy_sparse_matrix(result)



    #### Prune
    EPS = 1e-29#math.exp(-1.0/(2*_b) * math.log(1.0/_a * (1.0/EPS_0 -1)))
    print("Epsilon is " + str(EPS))
    print("Hyperparameters a = " + str(_a) + " and b = " + str(_b))



    model = GNN(data.num_features, hid_dim, dim, n_layers=n_layers,
                must_propagate=must_propagate,
                norm=norm)
    model = model.to(device)
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=wd)
    new_data = Data(x=data.x, edge_index=target_graph_index,
                    y=data.y, edge_attr=target_graph_weights)
    sparsity =  new_data.num_edges/(new_data.num_nodes**2 -new_data.num_nodes)
    if repulsion_strength is None:
        repulsion_strength = 1.0/sparsity
    row_pos, col_pos =  new_data.edge_index
    index = (row_pos != col_pos)
    edge_weights_pos = new_data.edge_attr#[index]

    if target is not None:
        edge_weights_pos = fast_intersection(row_pos[index], col_pos[index], edge_weights_pos,
                                             target, unknown_dist=1.0, far_dist=5.0)

    if subsampling is None:
        row_neg, col_neg = negative_sampling(new_data.edge_index, num_neg_samples = 5 * new_data.edge_index.shape[1] )
        index_neg = (row_neg != col_neg)
        edge_weights_neg = EPS * torch.ones(len(row_neg))
        if target is not None:
            edge_weights_neg = fast_intersection(row_neg[index_neg], col_neg[index_neg], edge_weights_neg,
                                                 target, unknown_dist=1.0, far_dist=5.0)
    best_t=0
    cnt_wait = 0
    best=1e9
    log_sigmoid = torch.nn.LogSigmoid()
    edges = [(e[0],e[1]) for _, e in enumerate(data.edge_index.numpy().T)]
    for epoch in range(epochs):
        tic_epoch = time.time()
        model.train()
        optimizer.zero_grad()
        tic = time.time()
        out = model(data.x.float(), data.edge_index)
        diff_norm = torch.sum(torch.square(out[row_pos[index]] - out[col_pos[index]]), 1)
        diff_norm = torch.clip(diff_norm, min=1e-3)
        log_q = -torch.log1p(_a *  diff_norm ** _b)
        loss_pos = - torch.mean(edge_weights_pos[index] * log_sigmoid(log_q)) - torch.mean((1. - edge_weights_pos[index]) *  (log_sigmoid(log_q) - log_q ) * repulsion_strength)

        if subsampling is None:
            diff_norm_neg = torch.sum(torch.square(out[row_neg[index_neg]] - out[col_neg[index_neg]]), 1) #+ 1e-3
            diff_norm_neg = torch.clip(diff_norm_neg, min=1e-3)
            log_q_neg = -torch.log1p(_a *  diff_norm_neg ** _b)
        else:
            row_neg, col_neg = negative_sampling(new_data.edge_index,
                                                 num_neg_samples=subsampling)
            index_neg = (row_neg != col_neg)
            edge_weights_neg = EPS * torch.ones(len(row_neg))
            if target is not None:
                edge_weights_neg = fast_intersection(row_neg[index_neg],
                                                     col_neg[index_neg], edge_weights_neg,
                                                     target, unknown_dist=1.0, far_dist=5.0)
            diff_norm_neg = torch.sum(torch.square(out[row_neg[index_neg]] - out[col_neg[index_neg]]), 1) #+ 1e-3
            diff_norm_neg = torch.clip(diff_norm_neg, min=1e-3)
            log_q_neg = torch.log1p(_a *  diff_norm_neg ** _b)
        loss_neg = - torch.mean((log_sigmoid(log_q_neg) - log_q_neg ) * repulsion_strength)
        ### Add a term to make sure that the features are learned independently
        c1 = torch.mm(out.T, out)
        c1 = c1 / out.shape[0]
        iden = torch.tensor(np.eye(out.shape[1])).to(device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss = loss_pos + loss_neg +  lambd_corr * loss_dec1
        tic =  time.time()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=4)
        optimizer.step()
        for g in optimizer.param_groups:
            g['lr'] = lr * (1.0 - (float(epoch) / float(epochs)))

        print('Epoch={:03d}, loss={:.4f}, time={:.4f}'.format(epoch, loss.item(),time.time()-tic_epoch))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), os.getcwd()  + '/results/best_gnumap_'
                                          + str(method) + '_neigh' + str(neighbours)
                                          + '_dim' + str(dim) + '_' + name_file +  '.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == patience and epoch>50:
            print('Early stopping at epoch {}!'.format(epoch))
            break
        #print("Time epoch after saving", time.time()-tic_epoch)
    #tracker.stop()
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(os.getcwd()  + '/results/best_gnumap_' +
                                     str(method) + '_neigh' + str(neighbours)
                                     + '_dim' + str(dim) + '_' + name_file + '.pkl'))
    return(model,target_graph_index)


def train_grace(data, channels, proj_hid_dim, n_layers=2, tau=0.5,
                epochs=100, wd=1e-5, lr=1e-3, fmr=0.2, edr =0.5,
                proj="nonlinear-hid", name_file="test", device=None,
                gnn_type="symmetric", alpha=0.5, beta = 1,add_self_loops= True, dropout_rate: float = 0.5):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        in_dim = data.num_features
        hid_dim = channels
        proj_hid_dim = proj_hid_dim
        n_layers = n_layers
        tau = tau
        N = data.num_nodes

        ##### Train GRACE model #####
        print("=== train GRACE model ===")
        model = GRACE(in_dim, hid_dim, proj_hid_dim, n_layers,dropout_rate, gnn_type, alpha , beta, add_self_loops, tau)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        #tracker = OfflineEmissionsTracker(country_iso_code="US", project_name='GRACE_'+ str(channels) +
        #            '_proj_hid_dim' + str(proj_hid_dim) + '_tau'+ str(tau) +
        #            '_edr' + str(edr) + '_fmr'  +str(fmr) + '_' + proj + '_' +  name_file)
        def train_grace_one_epoch(model, data, fmr, edr, proj):
                model.train()
                optimizer.zero_grad()
                new_data1,_ = random_aug(data, fmr, edr)
                new_data2,_ = random_aug(data, fmr, edr)
                new_data1 = new_data1.to(dev)
                new_data2 = new_data2.to(dev)
                z1, z2 = model(new_data1, new_data2)
                loss = model.loss(z1, z2, layer=proj)
                loss.backward()
                optimizer.step()
                return loss.item()
        #tracker.start()
        for epoch in range(epochs):
            loss = train_grace_one_epoch(model, data, fmr,
                                          edr, proj)
            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
        #tracker.stop()
        return(model)


def train_cca_ssg(data,  hid_dim, channels, lambd=1e-5,
                  n_layers=2, epochs=100, lr=1e-3,
                  fmr=0.2, edr =0.5, name_file="test",
                  device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_dim = data.num_features
    hid_dim =  hid_dim
    out_dim = channels
    N = data.num_nodes
    ##### Train the SelfGCon model #####
    print("=== train CCa model model ===")
    model = CCA_SSG(in_dim, hid_dim, out_dim, n_layers, lambd, N, use_mlp=False) #
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    #tracker = OfflineEmissionsTracker(country_iso_code="US", project_name='CCA-SSG_'+ str(channels) +
    #            '_lambda' + str(lambd) +
    #            '_edr' + str(edr) + '_fmr'  +str(fmr) + '_' +  name_file)

    def train_cca_one_epoch(model, data):
        model.train()
        optimizer.zero_grad()
        new_data1 = random_aug(data, fmr, edr)
        new_data2 = random_aug(data, fmr, edr)
        new_data1 = new_data1.to(device)
        new_data2 = new_data2.to(device)
        z1, z2 = model(new_data1, new_data2)
        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        return loss.item()
    #tracker.start()
    for epoch in range(epochs):
        loss = train_cca_one_epoch(model, data) #train_semi(model, data, num_per_class, pos_idx)
        # print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
    #tracker.stop()
    return(model)


def train_bgrl(data, channels, lambd=1e-5,
                  n_layers=2, epochs=100, lr=1e-3,
                  fmr=0.2, edr =0.5, pred_hid=512, wd=1e-5,
                  drf1=0.2, drf2=0.2, dre1=0.4, dre2=0.4, name_file="test",
                  device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = data.num_features
    hid_dim = channels
    out_dim = channels
    n_layers = n_layers

    num_class = int(data.y.max().item()) + 1
    N = data.num_nodes

    ##### Train the BGRL model #####
    print("=== train BGRL model ===")
    model = BGRL(in_dim, hid_dim, out_dim, n_layers, pred_hid)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay= wd)
    s = lambda epoch: epoch / 1000 if epoch < 1000 \
                    else ( 1 + np.cos((epoch-1000) * np.pi / (epochs - 1000))) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = s)
    #tracker = OfflineEmissionsTracker(country_iso_code="US", project_name='BGRL_'+ str(channels) +
    #            '_lambda' + str(lambd) +
    #            '_edr' + str(edr) + '_fmr'  +str(fmr) +
    #            '_drf1' + str(drf1) + '_dre1'  +str(dre1) + '_pred_hid' +
    #            str(pred_hid) + '_' +  name_file)


    def train_bgrl_one_epoch(model, data):
        model.train()
        optimizer.zero_grad()
        new_data1 = random_aug(data, drf1, dre1)
        new_data2 = random_aug(data, drf2, dre2)

        z1, z2, loss = model(new_data1, new_data2)

        loss.backward()
        optimizer.step()
        scheduler.step()
        model.update_moving_average()

        return loss.item()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = train_bgrl_one_epoch(model, data)
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
    return(model)

def train_vgnae():
    model = DeepVGAEX(data.x.size()[1], out_channels, out_channels,
                         n_layers=n_lay, normalize=args.normalize,
                         h_dims_reconstructiony = [out_channels, out_channels],
                         y_dim=alpha, dropout=0.5,
                         lambda_y =0.5/alpha, activation=args.non_linear).to(device)
    w = torch.randn(size= (data.num_features, alpha)).float()
    y_randoms = torch.mm(data.x, w)
    # move to GPU (if available)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    last_ac = 0
    trigger_times = 0
    best_epoch_model = 0
    temp_res = []


    for epoch in range(1, args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(train_data.x, y=y_randoms,
                          pos_edge_index=train_data.pos_edge_label_index,
                          neg_edge_index=train_data.neg_edge_label_index,
                          train_mask=train_data.train_mask)
        loss.backward()
        optimizer.step()
        #if epoch == 50: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/2)
        #if epoch == 100: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/4)
        #if epoch == 150: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/8)
        if epoch % 5 == 0:
            loss = float(loss)
            train_auc, train_ap = model.single_test(data.x,
                                train_data.pos_edge_label_index,
                                train_data.pos_edge_label_index,
                                train_data.neg_edge_label_index)
            roc_auc, ap = model.single_test(data.x,
                                train_data.pos_edge_label_index,
                                test_data.pos_edge_label_index,
                                test_data.neg_edge_label_index)
            temp_res += [[epoch, train_auc, train_ap, roc_auc, ap]]
            print('Epoch: {:03d}, LOSS: {:.4f}, AUC(train): {:.4f}, AP(train): {:.4f}  AUC(test): {:.4f}, AP(test): {:.4f}'.format(epoch, loss, train_auc, train_ap, roc_auc, ap))

            #### Add early stopping to prevent overfitting
            out  = model.single_test(data.x,
                                train_data.pos_edge_label_index,
                                val_data.pos_edge_label_index,
                                val_data.neg_edge_label_index)
            current_ac = np.mean(out)
    return(model)


def train_clgr(data,  hid_dim, channels,
                          n_layers=2, epochs=100, lr=1e-3,
                          tau=0.1, edr=0.2, fmr=0.2,
                          name_file="test", normalize=True,
                          standardize=True, patience=20,
                          device=None, mlp_use=False, lambd=1e-1, hinge=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_dim = data.num_features
    hid_dim =  hid_dim
    out_dim = channels
    N = data.num_nodes
    ##### Train the SelfGCon model #####
    print("=== train SelfGCon model ===")
    model = CLGR(in_dim, hid_dim, out_dim,
                 n_layers, tau, use_mlp = mlp_use,
                 normalize=normalize, standardize=standardize,
                  lambd=lambd, hinge=hinge) #
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    #tracker = OfflineEmissionsTracker(country_iso_code="US", project_name='CCA-SSG_'+ str(channels) +
    #            '_lambda' + str(lambd) +
    #            '_edr' + str(edr) + '_fmr'  +str(fmr) + '_' +  name_file)
    def train_clgr_one_epoch(model, data):
        model.train()
        optimizer.zero_grad()
        new_data1 = random_aug(data, fmr, edr)
        new_data2 = random_aug(data, fmr, edr)
        new_data1 = new_data1.to(device)
        new_data2 = new_data2.to(device)
        z1, z2 = model(new_data1, new_data2)
        loss = model.loss(z1, z2, device=device)
        loss.backward()
        optimizer.step()
        return loss.item()
    #tracker.start()

    best_t = 0
    cnt_wait = 0
    best = 1e9
    for epoch in range(epochs):
        loss = train_clgr_one_epoch(model, data) #train_semi(model, data, num_per_class, pos_idx)
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
        ### add patience
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), os.getcwd()  + '/results/best_clgr_'
                                           + name_file +  '.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == patience and epoch>50:
            print('Early stopping at epoch {}!'.format(epoch))
            break
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load( os.getcwd()  + '/results/best_clgr_'
                                           + name_file +  '.pkl'))
    #tracker.stop()
    return(model)