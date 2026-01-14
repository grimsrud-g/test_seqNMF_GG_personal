#!/usr/bin/env python3
import os
import sys
import numpy as np

# ---- user input ----
#in_path = "/scratch/groups/anishm/fmri/BRAINS_Test50subj600Epoch/tempflip/Cortex_seed_001_K_1_L_8_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.001_L1H_0.0_all.npz"
in_path = sys.argv[1]
L_expected = 7
K_expected = 12
old_mask_path = sys.argv[2]
new_mask_path = '/scratch/groups/anishm/fmri/HCP_tSNR/renorm_H_tflip/mask_indices.npy'
full_length=5124 # hard coded, that's two fs4 hemis

def main():
    # load everything
    old_mask=np.load(old_mask_path)
    new_mask=np.load(new_mask_path)

    # just a lil sanity check
    print(f"Old mask: {len(old_mask)} vertices omitted")
    print(f"New mask: {len(new_mask)} vertices omitted")
    # convert old mask to "keep" to then subsequently subtract from
    old_keep = np.ones(full_length, dtype=bool)
    old_keep[old_mask]= False
    old_kept_indices = np.where(old_keep)[0]
    # same for new
    new_keep = np.ones(full_length, dtype=bool)
    new_keep[new_mask]= False
    new_kept_indices = np.where(new_keep)[0]
    # this is just a checkpoint loadin
    ckpt = np.load(in_path, allow_pickle=True)
    if "W" not in ckpt:
        raise ValueError("NPZ missing key 'W'")
    W = ckpt["W"]

    # infer axis layout: find L and K axes
    shape = tuple(W.shape)
    try:
        #l_axes = [i for i, s in enumerate(shape) if s == L_expected]
        #k_axes = [i for i, s in enumerate(shape) if s == K_expected]
	# need to update because K happens to equal L
        l_axes = [1]
        k_axes = [2]	
        if not l_axes or not k_axes:
            raise ValueError(f"Cannot find axes matching L={L_expected} and K={K_expected} in W.shape={shape}")
        # pick the first unique pair
        L_ax = l_axes[0]
        K_ax = k_axes[0]
        if L_ax == K_ax:
            raise ValueError(f"Ambiguous axes for L and K (both map to axis {L_ax}) in W.shape={shape}")
        # spatial axis is the remaining one
        all_axes = {0,1,2}
        N_ax = list(all_axes - {L_ax, K_ax})
        if len(N_ax) != 1:
            raise ValueError(f"Could not resolve spatial axis from W.shape={shape}")
        N_ax = N_ax[0]
    except Exception as e:
        raise RuntimeError(f"Axis-detection failed: {e}")

    print(f"Loaded W with shape {shape} (N axis={N_ax}, L axis={L_ax}, K axis={K_ax})")

    # bring to canonical (N, L, K)
    W_NLK = np.transpose(W, axes=(N_ax, L_ax, K_ax))  # shape (N, L, K)
    N_old, L, K = W_NLK.shape
    # now we need to do our reconstruction walk-through: first make W-full w/r/t original fsaverage4 every-vertex-included
    W_full = np.zeros((full_length, L, K), dtype=W.dtype)
    # now port in group-level fit
    W_full[old_kept_indices, :, :] = W_NLK
    # now apply TSNR-masked mask
    W_NLK = W_full[new_kept_indices, :, :]
    
    # compute 50th percentile (median) across N for each (L, K)
    # NOTE: seqNMF/NMF is nonnegative; if yours has tiny negatives, switch to np.abs(W_NLK)
    thresh = np.median(W_NLK, axis=0, keepdims=True)  # shape (1, L, K)
    # zero values below the median (strictly below; equal values kept)
    pruned = W_NLK.copy()
    pruned[pruned < thresh] = 0.0

    # restore original axis order
    inv_axes = np.argsort((N_ax, L_ax, K_ax))
    W_pruned = np.transpose(pruned, axes=inv_axes)

    # write out: same keys, W replaced
    base, ext = os.path.splitext(in_path)
    out_path = f"{base}_halfprunedW{ext}"
    #out_path = f"{base}_2_3rds_prunedW{ext}"
    # collect everything to save
    out = {k: ckpt[k] for k in ckpt.files}
    out["W"] = W_pruned.astype(W.dtype, copy=False)

    np.savez(out_path, **out)
    print(f"Saved pruned W to: {out_path}")

if __name__ == "__main__":
    main()

