import numpy as np
import scipy as sp
from numba import jit, prange

import warnings
from tqdm import tqdm
import torch
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
# Helper function to convert between numpy arrays and tensors
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()

#from skimage.util.shape import view_as_windows as viewW

#def strided_indexing_roll(a, r):
#    # Concatenate with sliced to cover all rolls
#    a_ext = np.concatenate((a,a[:,:-1]),axis=1)
#
#    # Get sliding windows; use advanced-indexing to select appropriate ones
#    n = a.shape[1]
#    return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]

def test_significance(test_data, W, p = 0.05, num_null = None):
    
    # % Remove factors where there is obviously no sequence
    # % That is, W is empty, or has one feature (pixel / neuron) with >99.9% of the power
    idx_empty = ~W.any(axis=(0,2))
    idx_empty = ~W.any(axis=(0,2))
    Wflat2 = W.sum(-1)**2
    idx_empty |= (Wflat2).max(0) > 0.999*Wflat2.sum(0) # one pixel/neuron has >99.9% of the power

    W = W.copy()
    W = W[:, ~idx_empty, :]
    
    N, K, L = W.shape
    T = test_data.shape[1]
    
    if num_null is None:
        num_null = int(np.ceil(K/p)*2)
        
    # make num_null shifted datasets 
    skew_null = np.zeros( (K, num_null) )

    X = test_data.copy()
    X_allshift = np.array([np.roll(X, -l, axis=1) for l in range(L)])

    for n in tqdm(range(num_null), total=num_null):
        # Make a null dataset
        Wnull = np.zeros((N,K,L))
        for k in range(K):
            for ni in range(N):
                Wnull[ni, k] = np.roll(W[ni, k], np.random.randint(L), axis=-1)

        # Calculate WTX
        WTX = np.einsum('lnt,nkl -> kt', X_allshift, Wnull, optimize=True)
        skew_null[:,n] = sp.stats.skew(WTX, axis=1, bias=False) # bias=False corrects for statistical bias, according to the docs
    
    WTX = np.einsum('lnt,nkl -> kt', X_allshift, W)
    skew = sp.stats.skew(WTX, axis=1, bias=False) # bias=False corrects for statistical bias, according to the docs
    
    pvals = np.zeros(K)
    for k in range(K):
        #% Assign pvals from skewness
        pvals[k] = (1+sum(skew_null[k,:]>skew[k]))/(num_null + 1) # JK added +1 to denominator because it feels more correct
    
    all_pvals = np.zeros(idx_empty.size)
    all_pvals[idx_empty] = np.inf
    all_pvals[~idx_empty] = pvals
    pvals = all_pvals
    is_significant = pvals <= (p/K)

    return pvals, is_significant, skew, skew_null

@jit(nopython=True, cache=True)
def get_shapes(W, H, force_full=False):
    N = W.shape[0]
    T = H.shape[1]
    K = W.shape[1]
    L = W.shape[2]

    #trim zero padding along the L and K dimensions
    if not force_full:
        W_sum = W.sum(axis=0).sum(axis=1)
        H_sum = H.sum(axis=1)
        K = 1
        for k in np.arange(W.shape[1]-1, 0, -1):
            if (W_sum[k] > 0) or (H_sum[k] > 0):
                K = k+1
                break

        L = 2
        for l in np.arange(W.shape[2]-1, 2, -1):
            W_sum = W.sum(axis=1).sum(axis=0)
            if W_sum[l] > 0:
                L = l+1
                break

    return N, K, L, T

@jit(nopython=True, cache=True)
def trim_shapes(W, H, N, K, L, T):
    return W[:N, :K, :L], H[:K, :T]

@jit(nopython=True, cache=True)
def roll_2d_axis1(arr, shift):
    result = np.empty_like(arr)
    for i in range(arr.shape[0]):
        result[i] = np.roll(arr[i], shift)
    return result

@jit(nopython=True, cache=True)
def reconstruct(W, H):
    N, K, L, T = get_shapes(W, H, force_full=True)
    W, H = trim_shapes(W, H, N, K, L, T)

    H = np.hstack((np.zeros((K, L), dtype=np.float32), H, np.zeros((K, L), dtype=np.float32)))
    T += 2 * L
    X_hat = np.zeros((N, T), dtype=np.float32)

    for t in np.arange(L):
        #X_hat += np.dot(W[:, :, t], np.roll(H, t - 1, axis=1)) #JK: the original has an off-by-one error in the np.roll
        X_hat += np.dot(W[:, :, t], roll_2d_axis1(H, t))

    return X_hat[:, L:-L]


def shift_factors(W, H):
    warnings.simplefilter('ignore') #ignore warnings for nan-related errors

    N, K, L, T = get_shapes(W, H, force_full=True)
    W, H = trim_shapes(W, H, N, K, L, T)

    if L > 1:
        center = int(np.max([np.floor(L / 2), 1]))
        Wpad = np.concatenate((np.zeros([N, K, L]), W, np.zeros([N, K, L])), axis=2)

        for i in np.arange(K):
            temp = np.sum(np.squeeze(W[:, i, :]), axis=0)
            # return temp, temp
            try:
                cmass = int(np.max(np.floor(np.sum(temp * np.arange(1, L + 1)) / np.sum(temp)), axis=0))
            except ValueError:
                cmass = center
            Wpad[:, i, :] = np.roll(np.squeeze(Wpad[:, i, :]), center - cmass, axis=1)
            H[i, :] = np.roll(H[i, :], cmass - center, axis=0)

    return Wpad[:, :, L:-L], H


def compute_loadings_percent_power(V, W, H):
    N, K, L, T = get_shapes(W, H)
    W, H = trim_shapes(W, H, N, K, L, T)

    loadings = np.zeros(K)
    var_v = np.sum(np.power(V, 2))

    for i in np.arange(K):
        WH = reconstruct(np.reshape(W[:, i, :], [W.shape[0], 1, W.shape[2]]),
                         np.reshape(H[i, :], [1, H.shape[1]]))
        loadings[i] = np.divide(np.sum(np.multiply(2 * V.flatten(), WH.flatten()) - np.power(WH.flatten(), 2)), var_v)

    loadings[loadings < 0] = 0
    return loadings

def compute_loadings_percent_power_voxelwise(V, W, H):
    N, K, L, T = get_shapes(W, H)
    W, H = trim_shapes(W, H, N, K, L, T)

    loadings = np.zeros(N)
    var_v = np.sum(np.power(V, 2))

    WH = reconstruct(W, H)
    loadings = np.divide(np.sum(np.multiply(2 * V.flatten(), WH.flatten()) - np.power(WH.flatten(), 2), axis=-1), var_v)

    loadings[loadings < 0] = 0
    return loadings

def rec_rank1(W, H):
    N, K, L = W.shape
    padH = F.pad(H, (L-1, 0))
    Wrev = torch.flip(W, (2,))

    # TODO: figure out if there is a way you can use 'groups' to solve this more elegantly... 
    Xhat_rnk1 = torch.cat(
        [F.conv1d(padH[[i]].unsqueeze(0), Wrev[:, [i], :], padding='valid') for i in range(K) ])

    # Sanity check: 
    #hi = padH[0].unsqueeze(0).unsqueeze(0)
    #wi = Wrev[:,0,:].unsqueeze(1)
    #torch.allclose(F.conv1d(hi, wi, padding='valid'), Xhat_rnk1[0]) # --> True

    return Xhat_rnk1

def pix_r2(Y, Yh, mask):
    sse = ((Y - Yh)**2).sum(0)
    sst = ((Y - Y.mean(0))**2).sum(0)
    r2 = 1 - sse/sst
    r2_pix = fill_masked_component(r2, mask)
    r2_pix[~mask] = np.nan
    return r2_pix

def fill_masked_component(x, mask):
    img = np.zeros(mask.shape)
    r,c = np.where(mask)
    img[r,c] = x
    return img

def make_W_montage(W, num_rows, num_cols, mask, sigmas=(0.1, 1, 1), norm = True):
    N, K, L = W.shape
    nr, nc = mask.shape
    stack = None
    for i in range(num_rows):
        row = None
        for j in range(num_cols):
            k = num_cols*i + j
            stk = np.zeros( ( L, nr, nc ) )
            if k >= W.shape[1]:
                row = np.concatenate((row, np.zeros_like(stk)), axis=2)
                continue
            Wk = W[:, k].copy()
            for ll in range(L):
                stk[ll] = fill_masked_component(Wk[:,ll], mask)
            if sigmas is not None:
                stk = sp.ndimage.gaussian_filter(stk, sigmas)
            if norm:
                #stk/=stk.max()
                #stk-=stk.min(0)
                stk = (stk - stk.min())/(stk.max() - stk.min())
            if row is None:
                row = stk.copy()
            else:
                row = np.concatenate((row, stk), axis=2)
        if stack is None:
            stack = row.copy()
        else:
            stack = np.concatenate((stack, row), axis=1)
        
    return stack

def dissimilarity(W1, H1, W2, H2):
    """ TODO """
    if isinstance(W1, np.ndarray):
        W1 = to_t(W1)
    if isinstance(H1, np.ndarray):
        H1 = to_t(H1)
    if isinstance(W2, np.ndarray):
        W2 = to_t(W2)
    if isinstance(H2, np.ndarray):
        H2 = to_t(H2)

    K = H1.shape[0]
    C = np.zeros((K,K))
    Xhat_1 = rec_rank1(W1, H1)
    sx1 = torch.sqrt((Xhat_1**2).sum((1,2)))
    Xhat_2 = rec_rank1(W2, H2)
    sx2 = torch.sqrt((Xhat_2**2).sum((1,2)))
    C = torch.einsum('knt,jnt->kj', Xhat_1, Xhat_2)
    C /= torch.outer(sx1, sx2) + np.finfo(float).eps
    C = from_t(C)
    del Xhat_1, Xhat_2, sx1, sx2
    maxrow = C.max(0)
    maxcol = C.max(1)
    maxrow[np.isnan(maxrow)] = 0
    maxcol[np.isnan(maxcol)] = 0
    diss = 1/2/K*(2*K - maxrow.sum() - maxcol.sum())
    return diss

def dissimilarity_cpu(W1, H1, W2, H2):
    """ TODO """
    N, K, L = W1.shape
    K, T = H1.shape
    C = np.zeros((K,K), dtype=np.float32)
    Xh1_arr = np.zeros((K, N*T), dtype=np.float32)
    Xh2_arr = np.zeros((K, N*T), dtype=np.float32)
    sx1_arr = np.zeros(K, dtype=np.float32)
    sx2_arr = np.zeros(K, dtype=np.float32)
    for i in range(K):
        Xh1_arr[i] = reconstruct(W1[:,[i],:], H1[[i],:]).flatten()
        Xh2_arr[i] = reconstruct(W2[:,[i],:], H2[[i],:]).flatten()
        sx1_arr[i] = np.sqrt(Xh1_arr[i]@Xh1_arr[i])
        sx2_arr[i] = np.sqrt(Xh2_arr[i]@Xh2_arr[i])
    for i in range(K):
        Xh1 = Xh1_arr[i]
        sx1 = sx1_arr[i]
        for j in range(K):
            Xh2 = Xh2_arr[j]
            sx2 = sx2_arr[j]
            C[i,j] = Xh1 @ Xh2
            C[i,j] /= sx1*sx2 + np.finfo(float).eps
    maxrow = C.max(0)
    maxcol = C.max(1)
    maxrow[np.isnan(maxrow)] = 0
    maxcol[np.isnan(maxcol)] = 0
    diss = 1/2/K*(2*K - maxrow.sum() - maxcol.sum())
    return diss

@jit(nopython=True, cache=True)
def max_along_axis0(arr):
    rows, cols = arr.shape
    max_values = np.empty(cols, dtype=arr.dtype)
    for col in range(cols):
        max_val = arr[0, col]
        for row in range(1, rows):
            if arr[row, col] > max_val:
                max_val = arr[row, col]
        max_values[col] = max_val
    return max_values

@jit(nopython=True, parallel=True)
def dissimilarity_cpu_numba(W1, H1, W2, H2):
    """ TODO """
    N, K, L = W1.shape
    K, T = H1.shape
    C = np.zeros((K,K), dtype=np.float32)
    Xh1_arr = np.zeros((K, N*T), dtype=np.float32)
    Xh2_arr = np.zeros((K, N*T), dtype=np.float32)
    sx1_arr = np.zeros(K, dtype=np.float32)
    sx2_arr = np.zeros(K, dtype=np.float32)
    for i in prange(K):
        Xh1_arr[i] = reconstruct(W1[:,i:i+1,:], H1[i:i+1,:]).flatten()
        Xh2_arr[i] = reconstruct(W2[:,i:i+1,:], H2[i:i+1,:]).flatten()
        sx1_arr[i] = np.sqrt(Xh1_arr[i]@Xh1_arr[i])
        sx2_arr[i] = np.sqrt(Xh2_arr[i]@Xh2_arr[i])
    for i in prange(K):
        Xh1 = Xh1_arr[i]
        sx1 = sx1_arr[i]
        for j in range(i, K):
            Xh2 = Xh2_arr[j]
            sx2 = sx2_arr[j]
            C[i,j] = Xh1 @ Xh2
            C[i,j] /= sx1*sx2 + 2.220446049250313e-16 # np.finfo(float).eps
            C[j,i] = C[i,j]
            if np.isnan(C[i, j]):
                C[i, j] = 0.
                C[j, i] = 0.
    maxrow = max_along_axis0(C) # C.max(0)
    maxcol = max_along_axis0(C.T) # C.max(1)
    # maxrow[np.isnan(maxrow)] = 0
    # maxcol[np.isnan(maxcol)] = 0
    diss = 1/2/K*(2*K - maxrow.sum() - maxcol.sum())
    return diss

def get_proc_data(args):
    subj_dir = f'{args.group_dir}/{args.date}_{args.mouse}_{args.session}/'
    dataf = f'{subj_dir}/{args.in_fname}.npy'
    outf = f'{subj_dir}/{args.out_fname}'

    print(f'INFO: LOADING DATA FROM: {dataf}')
    X = np.load(dataf)
    if args.do_filter:
        print('INFO: FILTERING DATA')
        nyq = args.fps / 2
        cut = np.array(args.filt_freqs) / nyq
        b, a = sp.signal.butter(args.filt_order, cut, args.filt_type)
        X = sp.signal.filtfilt(b, a, X, axis=0)

    if X.min() < 0:
        print('WARNING: DATA HAVE NEGATIVE VALUES. SUBTRACTING OFF THE MIN')
        X-=X.min()

    if X.shape[1] < X.shape[0]:
        print('WARNING: DATA SHOULD BE N x T WITH T > N. TRANSPOSING!')
        X=X.T

    if args.zscore:
        print('INFO: Z-SCORING DATA PIXELWISE')
        X = sp.stats.zscore(X, axis=1)

    if args.rescale01:
        print('INFO: RESCALING DATA TO RANGE [0,1]')
        X = (X - X.min(1, keepdims=True)) / (X.max(1, keepdims=True) - X.min(1, keepdims=True))

    if np.isnan(X).any():
        print('WARNING: FOUND NANS IN X, SETTING TO ZERO')
        print(f'# bad pixels = {np.isnan(X).any(1).sum()} (/{X.shape[0]})') 
        X[np.isnan(X).any(1)] = 0

    return X
