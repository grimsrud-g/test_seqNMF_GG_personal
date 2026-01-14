#!/usr/bin/env python3
"""
convert_fs6_funcgii_to_npz.py
Convert fsaverage6 func.gii files back to .npz format with fs6 tSNR mask
"""
import numpy as np
import nibabel as nib
from pathlib import Path
import glob

# Parameters
nL_fs6 = nR_fs6 = 40962  # fsaverage6

# Load fs6 mask
mask_fs6_path = "/scratch/groups/anishm/fmri/HCP_tSNR/fs6/mask_indices_fs6.npy"
mask_fs6_indices = np.load(mask_fs6_path).astype(int)
valid_fs6 = np.setdiff1d(np.arange(nL_fs6 + nR_fs6), mask_fs6_indices)
keepL_fs6 = valid_fs6[valid_fs6 < nL_fs6]
keepR_fs6 = valid_fs6[valid_fs6 >= nL_fs6] - nL_fs6

print(f"FS6 Mask loaded:")
print(f"  Total vertices: {nL_fs6 + nR_fs6}")
print(f"  Masked vertices: {len(mask_fs6_indices)}")
print(f"  Valid vertices: {len(valid_fs6)} ({len(keepL_fs6)} L, {len(keepR_fs6)} R)")

def load_func_gii_to_3d(gii_path):
    """Load GIFTI file and convert back to 3D array (n_vertices, K, L)"""
    img = nib.load(gii_path)
    n_verts = img.darrays[0].data.shape[0]
    n_cols = len(img.darrays)
    
    # Reconstruct 2D array (vertices, K*L)
    arr2d = np.zeros((n_verts, n_cols), dtype=np.float32)
    for i, darray in enumerate(img.darrays):
        arr2d[:, i] = darray.data
    
    return arr2d, n_verts, n_cols

def process_funcgii_pair(LH_path, RH_path, out_npz_path):
    """Convert LH/RH func.gii pair to masked .npz"""
    print(f"\nProcessing: {Path(LH_path).name}")
    
    # Load both hemispheres
    LH_arr2d, n_verts_L, n_cols = load_func_gii_to_3d(LH_path)
    RH_arr2d, n_verts_R, _ = load_func_gii_to_3d(RH_path)
    
    print(f"  Loaded LH: {LH_arr2d.shape}, RH: {RH_arr2d.shape}")
    
    # Verify vertex counts
    assert n_verts_L == nL_fs6, f"Expected {nL_fs5} LH vertices, got {n_verts_L}"
    assert n_verts_R == nR_fs6, f"Expected {nR_fs5} RH vertices, got {n_verts_R}"
    
    # Apply mask to extract only valid vertices
    LH_masked = LH_arr2d[keepL_fs6, :]  # shape: (len(keepL_fs5), K*L)
    RH_masked = RH_arr2d[keepR_fs6, :]  # shape: (len(keepR_fs5), K*L)
    
    print(f"  After masking: LH={LH_masked.shape}, RH={RH_masked.shape}")
    
    # Concatenate hemispheres
    W_2d = np.vstack([LH_masked, RH_masked])  # shape: (len(valid_fs5), K*L)
    
    # Infer K and L from the number of columns
    # The original shape was (n_vertices, K, L) -> reshaped to (n_vertices, K*L)
    # We need to figure out K and L
    # This is tricky - we need to know K or L to reconstruct
    # Let's check if we can infer from common values
    
    # Common values: K=12, L=7 or similar
    # Let's try to infer K from the filename or use a default
    fname = Path(LH_path).stem
    
    # Try to parse K and L from filename
    import re
    k_match = re.search(r'K_(\d+)', fname)
    l_match = re.search(r'L_(\d+)', fname)
    
    if k_match and l_match:
        K = int(k_match.group(1))
        L = int(l_match.group(1))
        print(f"  Parsed from filename: K={K}, L={L}")
    else:
        # Default fallback
        K = 12
        L = n_cols // K
        print(f"  Could not parse K/L from filename, using K={K}, L={L}")
    
    # Verify
    assert K * L == n_cols, f"K*L={K*L} doesn't match n_cols={n_cols}"
    
    # Reshape back to 3D
    W_3d = W_2d.reshape(len(valid_fs6), K, L, order='C')
    print(f"  Final W shape: {W_3d.shape}")
    
    # Save as .npz
    np.savez(out_npz_path, W=W_3d)
    print(f"  ✓ Saved: {out_npz_path}")
    
    return W_3d.shape

def process_directory(in_dir):
    """Process all LH/RH func.gii pairs in a directory"""
    in_dir = Path(in_dir)
    out_dir = in_dir.parent / "fs6_npz"
    out_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Processing directory: {in_dir}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*80}")
    
    # Find all LH func.gii files
    lh_files = sorted(in_dir.glob("LH_*_fs6.func.gii"))
    
    if not lh_files:
        print("No LH_*_fs6.func.gii files found!")
        return
    
    print(f"Found {len(lh_files)} LH files to process")
    
    for lh_path in lh_files:
        # Find corresponding RH file
        rh_name = lh_path.name.replace("LH_", "RH_")
        rh_path = lh_path.parent / rh_name
        
        if not rh_path.exists():
            print(f"WARNING: Missing RH pair for {lh_path.name}, skipping")
            continue
        
        # Create output filename
        stem = lh_path.stem.replace("LH_", "").replace("_fs6", "")
        out_npz = out_dir / f"{stem}_fs6_masked.npz"
        
        # Skip if already exists
        if out_npz.exists():
            print(f"Output exists, skipping: {out_npz.name}")
            continue
        
        try:
            process_funcgii_pair(lh_path, rh_path, out_npz)
        except Exception as e:
            print(f"ERROR processing {lh_path.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Process the test directories
    test_dirs = [
        "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC01/0/fs5_npz/fs6_func_gii",
    ]
    
    for dir_path in test_dirs:
        if Path(dir_path).exists():
            process_directory(dir_path)
        else:
            print(f"WARNING: Directory not found: {dir_path}")
    
    print("\n" + "="*80)
    print("All directories processed!")
    print(f"\nTo use the new .npz files:")
    print(f"  nL = nR = {nL_fs6}")
    print(f"  mask_both = '{mask_fs6_path}'")
    print(f"  valid = np.setdiff1d(np.arange(nL + nR), np.load(mask_both).astype(int))")

