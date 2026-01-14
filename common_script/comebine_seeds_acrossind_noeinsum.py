import numpy as np

import sys, os
sys.path.append('..')
from src.helpers import shift_factors
from src.consensus import get_weighted_adj, get_louvain_partition, get_basis_motifs, compute_xcors_noeinsum, motif_motif_xcor_optimized
from numba_progress import ProgressBar

def read_Ws(file_list, W_all_file, recompute=False):
    if not recompute and os.path.exists(W_all_file):
        W_all = np.load(W_all_file)
    else:
        cnt = 0
        for file in file_list:
            try:
                results = np.load(file)
            except (FileNotFoundError, EOFError) as e:
                print(file, ' not found')
                continue

            try:
                W = results['W'] # shape: 3658*K*L
            except IndexError as e:
                W = results
            L = W.shape[2]
            W = np.transpose(W, (1, 0, 2))
            W_norm = W - W.mean((1, 2), keepdims=True)
            norms = np.linalg.norm(W_norm, axis=(1, 2))
            cnt += np.argwhere(norms != 0).shape[0]

        print(f'# of Ws: {cnt}')
        W_all = np.zeros((cnt, W.shape[1], L), dtype=np.float32)
        cnt = 0
        for file in file_list:
            try:
                results = np.load(file)
            except (FileNotFoundError, EOFError) as e:
                print(file, ' not found')
                continue
            
            try:
                W = results['W'] # shape: 3658*K*L
            except IndexError as e:
                W = results
            W = np.transpose(W, (1, 0, 2))
            W_norm = W - W.mean((1, 2), keepdims=True)
            norms = np.linalg.norm(W_norm, axis=(1, 2))
            W = W[norms != 0]
            W_all[cnt:cnt+W.shape[0]] = W
            cnt += W.shape[0]
        np.save(W_all_file, W_all)
    
    return W_all

def group_consensus_Ws(W_all, xcros_file, consensus_motifs_file, L, nnk, recompute=False):
    if not recompute and os.path.exists(xcros_file):
        xcors = np.load(xcros_file)
    else:
        print(xcros_file, ' not found')
        W_all_mean = W_all.mean((1, 2), keepdims=True)
        W_all -= W_all_mean
        xcors = motif_motif_xcor_optimized(W_all)
        # norms = np.linalg.norm(W_all, axis=(1,2))
        # xcors = np.zeros((W_all_mean.shape[0], W_all_mean.shape[0], L))

        # with ProgressBar(total=W_all.shape[0]) as numba_progress:
        #     compute_xcors_noeinsum(W_all, norms, xcors, numba_progress)
        W_all += W_all_mean

        np.save(xcros_file, xcors)

    peak_xcors = np.nanmax(xcors, axis=-1)
    peak_lags = xcors.argmax(-1) - (L  - 1)

    with ProgressBar(total=peak_xcors.shape[0]) as progress_bar:
        wgt_adj = get_weighted_adj(peak_xcors, progress_bar, nnk=nnk)
        
        
    print('Computing Louvain partition')
    labels = get_louvain_partition(wgt_adj)

    print('Computing consensus motifs')
    consensus_motifs = get_basis_motifs(labels, W_all, wgt_adj, peak_lags)

    # Finally, center the motifs:
    print('Centering motifs')
    fake_H = np.zeros((consensus_motifs.shape[0], 2*consensus_motifs.shape[-1]))
    consensus_motifs, _ = shift_factors(consensus_motifs.transpose(1,0,2), fake_H)
    np.save(f'{consensus_motifs_file}', consensus_motifs)
