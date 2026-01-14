import numpy as np
from tqdm import tqdm
from .helpers import get_proc_data
from tqdm import trange
from numba import jit, prange
from numba_progress import ProgressBar

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw_path

from skimage.util import view_as_windows
import networkx as nx
import community as community_louvain
from scipy.stats import zscore
import torch
import torch.nn as nn
import torch.nn.functional as F
# Specify that we want our tensors on the GPU and in float32
#device = torch.device('cuda')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
# Helper function to convert between numpy arrays and tensors
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()

def load_all_WH(args):
    subj_dir = f'{args.group_dir}/{args.date}_{args.mouse}_{args.session}'
    
    allW = []
    allH = []

    for i in tqdm(range(args.num_fits)):
        seq_file = f'{subj_dir}/{args.in_prefix}{i:03}.npz'
        try:
            data = np.load(seq_file)
            allW.append(data['W'])
            allH.append(data['H'])
        except:
            print(f'failed for file: {seq_file}')

    allH = np.array(allH)
    allW = np.array(allW)
    
    return allH, allW

@jit(nopython=True, parallel=True, cache=True)
def compute_xcors_lowmem(flatW, norms, start_idx, end_idx, peaks):
    # please make sure that the flatW has been mean-subtracted and padded
    L = int((flatW.shape[-1] + 1) / 3)

    xcors = np.zeros((flatW.shape[0], L*2), dtype=np.float32)

    for i in range(start_idx, end_idx):
        for j in prange(flatW.shape[0]):
            for k in range(2*L):
                xcors[j, k] = np.sum(flatW[i, :, L-1:2*L-1] * flatW[j, :, k:k+L]) / norms[i] / norms[j]
                if np.isnan(xcors[j, k]):
                    xcors[j, k] = 0
            peaks[0, i, j] = xcors[j, L//2:-L//2].max()
            peaks[1, i, j] = xcors[j, L//2:-L//2].argmax() - L
    
    return peaks

@jit(nopython=True, parallel=True)
def einsum(flatW, xcors, progress_hook):
    N_W, N_spatial, L = flatW.shape
    left_lag, right_lag = -L//2, L//2
    for m in prange(N_W): # N_W
        for i in range(N_W): # N_W
            for j in range(N_spatial): # N_spatial
                for iw, w in enumerate(range(left_lag, right_lag)): # L, time-lag
                    st = max(w, -w)
                    for ik, k in enumerate(range(st, L)): # L
                        xcors[m, i, iw] += flatW[m, j, ik] * flatW[i, j, k]
        progress_hook.update(1)

def compute_xcors_noeinsum(flatW, xnorms, xcors, progress_hook):
    # flatW shape: N_W * N_spatial * L
    # norms shape: N_W * N_W
    # xcors shape: N_W * N_W * L
    print('Calculating correlation matrix')
    einsum(flatW, xcors, progress_hook)
    xnorms = np.outer(xnorms, xnorms)
    xcors /= xnorms[..., None]
    return xcors

@jit(nopython=True, parallel=True)
def motif_motif_xcov(flatW_0mean, pad_flatW_0mean, progress_hook=None):
    N,K,L = flatW_0mean.shape
    xcovs = np.zeros((N,N,2*L-1)) #, dtype=flatW_0mean.dtype)
    for i in prange(N): # motif1
        for j in range(i,N): #motif2 (main diagonal + upper triangle)
            for k in range(K): #spatial
                for t in range(2*L-1): #window
                    for l in range(L): #frame within window
                        xcovs[i,j,t] += flatW_0mean[i,k,l]*pad_flatW_0mean[j,k,-(L+t)+l]
        for j in range(i+1,N): #motif2 (lower triangle exploiting symmetry)
           for t in range(2*L-1):
               xcovs[j,i,-(t+1)] = xcovs[i,j,t]
        if progress_hook is not None:
            progress_hook.update(1)

    return xcovs
        
def motif_motif_xcor_optimized(flatW):
    N, K, L = flatW.shape
    flatW_0mean = np.ascontiguousarray(flatW - flatW.mean((1,2),keepdims=True))
    pad_flatW_0mean = np.ascontiguousarray(np.pad(
        flatW_0mean, 
        ((0,0),(0,0), (L-1, L-1))
    ))
    N = flatW.shape[0]
    with ProgressBar(total=N) as pbar:
        xcovs = motif_motif_xcov(flatW_0mean, pad_flatW_0mean, pbar)
    
    norms = np.linalg.norm(flatW_0mean, axis=(1,2))
    xnorms = np.outer(norms, norms)
    xcors = xcovs / xnorms[..., None]
    return xcors

def compute_xcors(flatW):
    L = flatW.shape[-1]
    
    pad_flatW = np.pad(
        flatW - flatW.mean((1,2), keepdims=True), 
        ((0,0),(0,0), (L-1, L))
    )
    print('pad_flatW shape', pad_flatW.shape, 'flatW shape', flatW.shape)

    win_pad_flatW = view_as_windows(
        pad_flatW.copy(), 
        pad_flatW.shape[:2] + (L,)
    ).squeeze()[::-1].transpose(1,2,3,0)
    print('win_pad_flatW shape', win_pad_flatW.shape)

    xcovs = np.einsum('mjk, ijkw -> miw', flatW - flatW.mean((1,2), keepdims=True), win_pad_flatW, optimize=True)
    print('xcovs shape', xcovs.shape)
    
    norms = np.linalg.norm(pad_flatW, axis=(1,2))
    print('norms shape', norms.shape)
    xnorms = np.outer(norms, norms)
    print('xnorms shape', xnorms.shape)
    xcors = xcovs / xnorms[..., None]
    xcors = xcors[...,L//2:-L//2]
    
    return xcors

def compute_xcors_torch(flatW, pad_len):
    
    flatW = to_t(flatW)
    padW = F.pad(flatW, (pad_len,pad_len,0,0,0,0))

    mean_kern = np.ones(flatW[0].shape)
    n = mean_kern.size
    mean_kern /= n
    mean_kern = to_t(mean_kern)
    win_means = F.conv1d(padW, mean_kern.unsqueeze(0), padding='valid')

    eXY = F.conv2d(
        to_t(padW).unsqueeze(1), 
        to_t(flatW).unsqueeze(1), 
        stride=1
    ).squeeze() / flatW[0].numel()

    eX = win_means
    eY = flatW.mean((1,2)).squeeze()[None,:,None]
    eXeY = eX*eY

    eX2 = F.conv1d(padW**2, mean_kern.unsqueeze(0), padding='valid')
    eY2 = (flatW**2).mean((1,2))[None, :, None]

    sX = torch.sqrt( eX2 - eX**2 )
    sY =  torch.sqrt( eY2 - eY**2 ) 
    sXsY = sX*sY

    torch_xc = ( eXY - eXeY ) / (sXsY + torch.finfo(float).eps)
    
    return from_t(torch_xc)

def compute_dtw_dist(flatW):
    # flatW shape: N_W * N_pts * L
    scaler = TimeSeriesScalerMeanVariance()
    flatW = scaler.fit_transform(flatW)
    dtw_dis_arr = np.zeros((flatW.shape[0], flatW.shape[0]), dtype=np.float32)
    for i in range(dtw_dis_arr.shape[0]):
        for j in range(i, dtw_dis_arr.shape[1]):
            _, dis = dtw_path(flatW[i], flatW[j])
            dtw_dis_arr[i, j] = dtw_dis_arr[j, i] = dis

    return dtw_dis_arr 

def get_basis_motifs_dtw(labels, all_motifs, weights):
    M, K, L = all_motifs.shape
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    all_bases = []
    #all_bases_wgt = []

    for l in unique_labels:
        in_cluster = labels == l
        cluster_motifs = all_motifs[in_cluster].copy()
        #cluster_motifs /= np.linalg.norm(cluster_motifs, axis=(1,2))[:,None,None]
        mins = cluster_motifs.min((1,2), keepdims=True)
        maxs = cluster_motifs.max((1,2), keepdims=True)

        cluster_motifs-=mins
        cluster_motifs/=maxs-mins

        # get the template cluster (here we'll use module degree zscore on the weighted graph)

        # TODO: scale back and use binarized adjacency if this sucks
        cluster_deg = weights[in_cluster,:][:,in_cluster].sum(0)
        cluster_degz = zscore(cluster_deg)
        template_idx = cluster_degz.argmax()
        #... i don't think it matters too too much... gonna keep this one bc i kinda like the idea but 
        # maybe its bad idk

        template_motif = cluster_motifs[template_idx]
        cluster_idx = np.where(in_cluster)[0]

        # Apply DTW for alignment
        aligned_cluster = []
        for series in cluster_motifs:
            path, _ = dtw_path(template_motif.T, series.T)
            path = np.array(path)
            if path.shape[0] == series.shape[1]:
                aligned_series = series[:, path[:, 1]]
            else:
                aligned_series = np.zeros(series.shape, dtype=np.float32)
                for i in range(series.shape[0]):
                    try:
                        aligned_series[:, i] = np.mean(series[:, path[np.argwhere(path[:, 0] == i)[..., 0]], 1], axis=-1)
                    except IndexError: 
                        continue
            aligned_cluster.append(aligned_series)

        all_bases.append(np.mean(aligned_cluster, axis=0))
        
    return np.array(all_bases)


@jit(nopython=True, parallel=True)
def calc_wgt_adj(peak_xcors, w_knn, progress_hook):
    wgt_adj = np.zeros_like(peak_xcors)
    tot = peak_xcors.shape[0]

    for i in prange(tot):
        for j in range(tot):
            wgt_adj[i,j] = (w_knn[i]*w_knn[j]).sum() / (w_knn[i]|w_knn[j]).sum()
        progress_hook.update(1)

    return wgt_adj

def get_weighted_adj(peak_xcors, progress_hook, nnk = 15):

    ranks = np.argsort(np.argsort(-peak_xcors*(1-np.eye(peak_xcors.shape[0])), axis=1))
    w_knn = ranks < nnk

    wgt_adj = calc_wgt_adj(peak_xcors, w_knn, progress_hook)

    return wgt_adj

def get_louvain_partition(wgt_adj):
    G = nx.convert_matrix.from_numpy_array(wgt_adj)
    partition = community_louvain.best_partition(G)
    lblG = np.array(list(partition.values()))
    uG, cntG = np.unique(lblG, return_counts=True)
    print(uG, cntG)
    
    return lblG

def get_basis_motifs(labels, all_motifs, weights, peak_lags, return_all=False, avg_percent=1.):
    M, K, L = all_motifs.shape
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    all_bases = []
    if return_all:
        all_bases_ind = list()
    #all_bases_wgt = []

    for l in unique_labels:
        in_cluster = labels == l
        cluster_motifs = all_motifs[in_cluster].copy()
        #cluster_motifs /= np.linalg.norm(cluster_motifs, axis=(1,2))[:,None,None]
        mins = cluster_motifs.min((1,2), keepdims=True)
        maxs = cluster_motifs.max((1,2), keepdims=True)

        cluster_motifs-=mins
        cluster_motifs/=maxs-mins

        # pad w/ zeros so we can shift and average
        cluster_motifs = np.pad(cluster_motifs, ((0,0), (0,0), (L,L)))
        # get the template cluster (here we'll use module degree zscore on the weighted graph)

        # TODO: scale back and use binarized adjacency if this sucks
        cluster_deg = weights[in_cluster,:][:,in_cluster].sum(0)
        cluster_degz = zscore(cluster_deg)
        template_idx = cluster_degz.argmax()
        #... i don't think it matters too too much... gonna keep this one bc i kinda like the idea but 
        # maybe its bad idk

        # OK, JUST WORKING W/ ADJACENCY
        #template_idx = adjacency[in_cluster,:][:,in_cluster].sum(0).argmax()

        #Or... GET MOST ZERO-LAG XC w/i cluster
        #cluster_zero_lag_xc = xcors[in_cluster,...][:,in_cluster,:][...,15]

        template_motif = cluster_motifs[template_idx]
        cluster_idx = np.where(in_cluster)[0]

        lags2template = peak_lags[cluster_idx[template_idx],:][cluster_idx]
        cluster_motifs_shifted = np.array(
            [np.roll(w, s, axis=-1) for w, s in zip(cluster_motifs, lags2template)]
        )

        avg_indices = np.argsort(cluster_degz)[::-1][:int(in_cluster.shape[0]*avg_percent)]
        basis_motif = cluster_motifs_shifted[avg_indices].mean(0)
        #movW = np.array( [ fill_masked_component(wi, allen_mask_ds) for wi in basis_motif.T ] )

        # take degree-weighted average
        #deg_wgts = cluster_deg / cluster_deg.sum()
        #basis_motif_wgt = (deg_wgts[:,None,None] * cluster_motifs_shifted).sum(0)
        #movWwgt = np.array( [ fill_masked_component(wi, allen_mask_ds) for wi in basis_motif_wgt.T ] )

        all_bases.append(basis_motif)
        if return_all:
            all_bases_ind.append(cluster_motifs_shifted)
        #all_bases_wgt.append(basis_motif_wgt)
        
    if return_all:
        return np.array(all_bases), all_bases_ind
    return np.array(all_bases)

def refit_data_consW(consensus_motifs, args):
    data = np.load(f'{args.group_dir}/{args.date}_{args.mouse}_{args.session}/{args.seqs_fname}.npz')
    Y = get_proc_data(args)
    #Y = np.load(f'{group_dir}/{date}_{mouse}_{sess}/Yai_ds.npy')
    #Y-=Y.min()
    
    K = data['K'].flatten()
    L = data['L'].flatten()[0]
    lam = data['lam'].flatten()[0]
    lambda_OrthH = data['lambda_OrthH'].flatten()[0]

    W_init = to_t(consensus_motifs.transpose(1,0,2))

    torch.manual_seed(args.rng_seed)
    model0 = SeqNMF(
        K=consensus_motifs.shape[0],
        L=L,lam=lam, 
        lambda_OrthH=lambda_OrthH,
        W_init=W_init, 
        W_fixed=True
    ).to(device)
    
    tY = to_t(np.pad(Y, ((0,0), (L,L)))).unsqueeze(0)

    if not model0.W_initialized:
        print('initw')
        model0.initialize_W(tY)

    if not model0.H_initialized:
        model0.initialize_H(tY)

    cost = np.zeros((args.num_iters))
    cost[0] = model0.compute_cost(tY)

    with torch.no_grad():
        for i in trange(1, args.num_iters, desc=f'fitting', leave=True):
            model0.do_mult_update_step(tY)
            cost[i] = model0.compute_cost(tY)

    Yhat, _, _ = model0.finalize()
    return Yhat.T, Y.T, from_t(model0.W), from_t(model0.H)
