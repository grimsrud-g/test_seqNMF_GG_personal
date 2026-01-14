#!/usr/bin/env python3
"""
create_mask_from_tsnr_fs5.py
Create mask indices from tSNR dscalar (mask vertices with tSNR < 62)
"""
import numpy as np
import nibabel as nib

# Parameters
nL = nR = 10242  # fsaverage5
tsnr_threshold = 62.0  # fixed threshold matching original 15th percentile

# Input tSNR file (fsaverage5)
tsnr_dscalar = "/scratch/groups/anishm/fmri/HCP_tSNR/HCP_PreProcd_group_average_tsnr_fs5.dscalar.nii"

# Load tSNR data
print(f"Loading {tsnr_dscalar}...")
cifti_img = nib.load(tsnr_dscalar)
tsnr_data_all = cifti_img.get_fdata()  # shape: (n_volumes, n_vertices)

print(f"Full tSNR data shape: {tsnr_data_all.shape}")

# Extract first volume only (the actual tSNR values)
tsnr_data = tsnr_data_all[0, :]  # shape: (20484,) = 10242*2

print(f"Using first volume only")
print(f"tSNR data shape: {tsnr_data.shape}")
print(f"tSNR range: [{tsnr_data.min():.2f}, {tsnr_data.max():.2f}]")
print(f"tSNR mean: {tsnr_data.mean():.2f}")
print(f"tSNR median: {np.median(tsnr_data):.2f}")

# Create mask: vertices with tSNR < 62 are masked out
mask_indices = np.where(tsnr_data < tsnr_threshold)[0]

print(f"\nUsing threshold: {tsnr_threshold}")
print(f"Masked vertices: {len(mask_indices)} out of {len(tsnr_data)} ({100*len(mask_indices)/len(tsnr_data):.2f}%)")

# Save mask indices
out_dir = "/scratch/groups/anishm/fmri/HCP_tSNR/fs5"
from pathlib import Path
Path(out_dir).mkdir(exist_ok=True, parents=True)

mask_file = f"{out_dir}/mask_indices_fs5.npy"
np.save(mask_file, mask_indices)
print(f"Saved mask to: {mask_file}")

# Create convenience arrays (like your original code)
valid = np.setdiff1d(np.arange(nL + nR), mask_indices)
keepL = valid[valid < nL]
keepR = valid[valid >= nL] - nL  # shift to RH local index

print(f"\nValid vertices: {len(valid)} total ({len(keepL)} L, {len(keepR)} R)")

# Save keep indices
np.save(f"{out_dir}/keepL_indices_fs5.npy", keepL)
np.save(f"{out_dir}/keepR_indices_fs5.npy", keepR)
print(f"Saved keepL and keepR indices")

# Print some statistics about the masking
print(f"\ntSNR statistics:")
print(f"  Masked vertices: tSNR < {tsnr_threshold:.2f}")
print(f"  Valid vertices: tSNR >= {tsnr_threshold:.2f}")
print(f"  Valid tSNR range: [{tsnr_data[valid].min():.2f}, {tsnr_data[valid].max():.2f}]")
print(f"  Valid tSNR mean: {tsnr_data[valid].mean():.2f}")

print("\nTo use this mask in your code:")
print(f'mask_both = "{mask_file}"')
print(f'nL = nR = {nL}')
print(f'valid = np.setdiff1d(np.arange(nL + nR), np.load(mask_both).astype(int))')
