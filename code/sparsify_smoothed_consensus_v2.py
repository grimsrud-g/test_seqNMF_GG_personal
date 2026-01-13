#!/usr/bin/env python3
import os
import sys
import numpy as np
import subprocess
import tempfile
import nibabel as nib

# ---- user input ----
in_path = sys.argv[1]
L_expected = 7
K_expected = 8
old_mask_path = '/scratch/groups/anishm/fmri/HCP_Test100subj600Epoch/renorm_W_norm/mask_indices.npy'
new_mask_path = '/scratch/groups/anishm/fmri/HCP_tSNR/renorm_H_tflip/mask_indices.npy'
full_length = 5124  # hard coded, that's two fs4 hemis

# Smoothing parameters
SMOOTH_FWHM = 6.0  # mm - adjust as needed
# Paths to surface files (fsaverage4)
SURF_L = '/oak/stanford/groups/anishm/Adam/TemplateBrains/humans/surf/resample_fsaverage/fsaverage4_std_sphere.L.3k_fsavg_L.surf.gii'
SURF_R = '/oak/stanford/groups/anishm/Adam/TemplateBrains/humans/surf/resample_fsaverage/fsaverage4_std_sphere.R.3k_fsavg_R.surf.gii'

def smooth_cortical_surface(data_full, surf_l, surf_r, fwhm):
    """
    Smooth data on cortical surface using workbench.

    Parameters:
    -----------
    data_full : array (5124, K, L) - full fsaverage4 space
    surf_l, surf_r : paths to left/right surface GIFTI files
    fwhm : smoothing kernel FWHM in mm

    Returns:
    --------
    smoothed : array (5124, K, L)
    """
    N, K, L = data_full.shape
    n_verts_l = 2562  # fsaverage4 left hemi vertices
    n_verts_r = 2562  # fsaverage4 right hemi vertices

    smoothed_full = np.zeros_like(data_full)

    # Process each (L, K) frame separately
    for k_idx in range(K):
        for l_idx in range(L):
            frame = data_full[:, k_idx, l_idx]

            # Split into hemispheres
            frame_l = frame[:n_verts_l]
            frame_r = frame[n_verts_l:]

            # Smooth left hemisphere
            smoothed_l = smooth_hemisphere(frame_l, surf_l, fwhm)

            # Smooth right hemisphere
            smoothed_r = smooth_hemisphere(frame_r, surf_r, fwhm)

            # Recombine
            smoothed_full[:n_verts_l, k_idx, l_idx] = smoothed_l
            smoothed_full[n_verts_l:, k_idx, l_idx] = smoothed_r

            print(f" Smoothed frame K={k_idx+1}/{K}, L={l_idx+1}/{L}")

    return smoothed_full

def smooth_hemisphere(data, surf_path, fwhm):
    """
    Smooth one hemisphere using wb_command.

    Parameters:
    -----------
    data : array (n_vertices,)
    surf_path : path to surface GIFTI
    fwhm : smoothing FWHM in mm

    Returns:
    --------
    smoothed : array (n_vertices,)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write input metric
        input_metric = os.path.join(tmpdir, 'input.func.gii')
        output_metric = os.path.join(tmpdir, 'output.func.gii')

        # Create GIFTI metric file
        img = nib.gifti.GiftiImage()
        darray = nib.gifti.GiftiDataArray(
            data=data.astype(np.float32),
            intent='NIFTI_INTENT_NONE',
            datatype='NIFTI_TYPE_FLOAT32'
        )
        img.add_gifti_data_array(darray)
        nib.save(img, input_metric)

        # Run wb_command smoothing
        cmd = [
            'wb_command', '-metric-smoothing',
            surf_path,
            input_metric,
            str(fwhm),
            output_metric
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"wb_command failed: {result.stderr}")

        # Read smoothed output
        smoothed_img = nib.load(output_metric)
        smoothed_data = smoothed_img.darrays[0].data

        return smoothed_data

def save_as_giftis(data_full, output_dir, n_verts_l=2562):
    """
    Save full surface data as GIFTI files (one left, one right, with all K×L frames stacked).

    Parameters:
    -----------
    data_full : array (5124, K, L)
    output_dir : directory to save GIFTIs
    n_verts_l : number of left hemisphere vertices
    """
    os.makedirs(output_dir, exist_ok=True)
    N, K, L = data_full.shape

    # Create left and right hemisphere GIFTIs
    img_l = nib.gifti.GiftiImage()
    img_r = nib.gifti.GiftiImage()
    
    # Stack all frames: K0L0, K0L1, ..., K0L6, K1L0, K1L1, ..., K1L6, etc.
    for k_idx in range(K):
        for l_idx in range(L):
            frame = data_full[:, k_idx, l_idx]
            
            # Add left hemisphere frame
            darray_l = nib.gifti.GiftiDataArray(
                data=frame[:n_verts_l].astype(np.float32),
                intent='NIFTI_INTENT_NONE',
                datatype='NIFTI_TYPE_FLOAT32'
            )
            img_l.add_gifti_data_array(darray_l)
            
            # Add right hemisphere frame
            darray_r = nib.gifti.GiftiDataArray(
                data=frame[n_verts_l:].astype(np.float32),
                intent='NIFTI_INTENT_NONE',
                datatype='NIFTI_TYPE_FLOAT32'
            )
            img_r.add_gifti_data_array(darray_r)
    
    # Save files
    out_l = os.path.join(output_dir, 'all_factors.L.func.gii')
    out_r = os.path.join(output_dir, 'all_factors.R.func.gii')
    nib.save(img_l, out_l)
    nib.save(img_r, out_r)
    
    print(f"Saved 2 GIFTI files with {K*L} frames each to: {output_dir}")


def main():
    # load everything
    old_mask = np.load(old_mask_path)
    new_mask = np.load(new_mask_path)

    # just a lil sanity check
    print(f"Old mask: {len(old_mask)} vertices omitted")
    print(f"New mask: {len(new_mask)} vertices omitted")

    # convert old mask to "keep" to then subsequently subtract from
    old_keep = np.ones(full_length, dtype=bool)
    old_keep[old_mask] = False
    old_kept_indices = np.where(old_keep)[0]

    # same for new
    new_keep = np.ones(full_length, dtype=bool)
    new_keep[new_mask] = False
    new_kept_indices = np.where(new_keep)[0]

    # this is just a checkpoint loadin
    ckpt = np.load(in_path, allow_pickle=True)
    if "W" not in ckpt:
        raise ValueError("NPZ missing key 'W'")
    W = ckpt["W"]

    # infer axis layout: find L and K axes
    shape = tuple(W.shape)
    try:
        l_axes = [2]
        k_axes = [1]
        if not l_axes or not k_axes:
            raise ValueError(f"Cannot find axes matching L={L_expected} and K={K_expected} in W.shape={shape}")

        L_ax = l_axes[0]
        K_ax = k_axes[0]
        if L_ax == K_ax:
            raise ValueError(f"Ambiguous axes for L and K (both map to axis {L_ax}) in W.shape={shape}")

        # spatial axis is the remaining one
        all_axes = {0, 1, 2}
        N_ax = list(all_axes - {L_ax, K_ax})
        if len(N_ax) != 1:
            raise ValueError(f"Could not resolve spatial axis from W.shape={shape}")
        N_ax = N_ax[0]
    except Exception as e:
        raise RuntimeError(f"Axis-detection failed: {e}")

    print(f"Loaded W with shape {shape} (N axis={N_ax}, L axis={L_ax}, K axis={K_ax})")

    # bring to canonical (N, K, L)
    W_NKL = np.transpose(W, axes=(N_ax, K_ax, L_ax))
    N_old, K, L = W_NKL.shape

    # reconstruct to full fsaverage4 space
    W_full = np.zeros((full_length, K, L), dtype=W.dtype)
    W_full[old_kept_indices, :, :] = W_NKL

    # ===== SMOOTH ON SURFACE =====
    print("\nSmoothing on cortical surface...")
    W_full_smoothed = smooth_cortical_surface(W_full, SURF_L, SURF_R, SMOOTH_FWHM)
    print("Smoothing complete!")

    # apply TSNR-masked mask
    W_NKL_smoothed = W_full_smoothed[new_kept_indices, :, :]

    # compute 50th percentile (median) across N for each (K, L)
    thresh = np.median(W_NKL_smoothed, axis=0, keepdims=True)  # shape (1, K, L)

    # zero values below the median
    W_NKL_pruned = W_NKL_smoothed.copy()
    W_NKL_pruned[W_NKL_pruned < thresh] = 0.0

    # reconstruct pruned data to full space for GIFTI export
    W_full_pruned = np.zeros((full_length, K, L), dtype=W.dtype)
    W_full_pruned[new_kept_indices, :, :] = W_NKL_pruned

    # ===== SAVE AS GIFTI FILES =====
    base_name = os.path.basename(in_path).replace('.npz', '')
    gifti_dir = os.path.join(os.path.dirname(in_path), f"{base_name}_smoothed6_halfpruned_giftis")
    print("\nSaving as GIFTI files...")
    save_as_giftis(W_full_pruned, gifti_dir)

    # ===== SAVE AS NPZ =====
    # restore original axis order for NPZ
    inv_axes = np.argsort((N_ax, K_ax, L_ax))
    W_pruned = np.transpose(W_NKL_pruned, axes=inv_axes)

    base, ext = os.path.splitext(in_path)
    out_path = f"{base}_smoothed6_halfprunedW{ext}"

    # collect everything to save
    out = {k: ckpt[k] for k in ckpt.files}
    out["W"] = W_pruned.astype(W.dtype, copy=False)
    np.savez(out_path, **out)

    print(f"\nSaved smoothed+pruned W to: {out_path}")

if __name__ == "__main__":
    main()

