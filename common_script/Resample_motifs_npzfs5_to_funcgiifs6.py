#!/usr/bin/env python3
"""
upsample_npz_fs5_to_fs6.py
Convert .npz solutions from fsaverage4 to fsaverage5 func.gii files
"""
import numpy as np
import nibabel as nib
import subprocess
from pathlib import Path
import sys

# Parameters
nL_fs4 = nR_fs4 = 2562  # fsaverage4
nL_fs5 = nR_fs5 = 10242  # fsaverage5
nL_fs6 = nR_fs6 = 40962  # fsaverage6

# fs5 mask
mask_fs5_path = "/oak/stanford/groups/anishm/fMRI_datasets/mask_indices_fs5.npy"
mask_fs5_indices = np.load(mask_fs5_path).astype(int)
valid_fs5 = np.setdiff1d(np.arange(nL_fs5 + nR_fs5), mask_fs5_indices)
keepL_fs5 = valid_fs5[valid_fs5 < nL_fs5]
keepR_fs5 = valid_fs5[valid_fs5 >= nL_fs5] - nL_fs5

# Template paths
template_base = "/oak/stanford/groups/anishm/Adam/TemplateBrains/humans/surf/resample_fsaverage"
sphereL_fs5 = f"{template_base}/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii"
sphereR_fs5 = f"{template_base}/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii"
areaL_fs5 = f"{template_base}/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii"
areaR_fs5 = f"{template_base}/fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii"
sphereL_fs6 = f"{template_base}/fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii"
sphereR_fs6 = f"{template_base}/fsaverage6_std_sphere.R.41k_fsavg_R.surf.gii"
areaL_fs6 = f"{template_base}/fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii"
areaR_fs6 = f"{template_base}/fsaverage6.R.midthickness_va_avg.41k_fsavg_R.shape.gii"

def save_func_gii(arr3d, out_path):
    """Save 3D array as GIFTI with multiple columns"""
    n, K, L = arr3d.shape
    arr2d = arr3d.reshape(n, K * L, order="C")
    darrays = [nib.gifti.GiftiDataArray(arr2d[:, i].astype(np.float32))
               for i in range(arr2d.shape[1])]
    nib.save(nib.gifti.GiftiImage(darrays=darrays), out_path)

def resample_gii(in_gii, out_gii, sphere_src, sphere_tgt, area_src, area_tgt):
    """Resample GIFTI from fsaverage4 to fsaverage5"""
    cmd = [
        "wb_command", "-metric-resample",
        str(in_gii), sphere_src, sphere_tgt,
        "ADAP_BARY_AREA", str(out_gii),
        "-area-metrics", area_src, area_tgt
    ]
    subprocess.run(cmd, check=True)

def process_npz(npz_path, out_dir):
    """Process a single .npz file"""
    npz_path = Path(npz_path)
    print(f"\n{'='*80}")
    print(f"Processing: {npz_path.name}")
    print(f"{'='*80}")
    
    # Load npz
    npz = np.load(npz_path)
    W = npz["W"].astype(np.float32)  # shape: (len(valid), K, L)
    K, L = W.shape[1], W.shape[2]
    print(f"W shape: {W.shape} (vertices, K={K}, L={L})")
    
    # Create temp directory
    temp_dir = out_dir / "temp_resample"
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    # Split masked W into LH and RH
    WLm = W[:len(keepL_fs5), :, :]
    WRm = W[len(keepL_fs5):, :, :]
    print(f"Split: LH={WLm.shape}, RH={WRm.shape}")
    
    # Expand to full fsaverage5 surfaces (with zeros for masked vertices)
    WL_fs5 = np.zeros((nL_fs5, K, L), np.float32)
    WR_fs5 = np.zeros((nR_fs5, K, L), np.float32)
    WL_fs5[keepL_fs5, :, :] = WLm
    WR_fs5[keepR_fs5, :, :] = WRm
    print(f"Expanded to fs4: LH={WL_fs5.shape}, RH={WR_fs5.shape}")
    
    # Save as fsaverage4 func.gii
    LH_fs5_gii = temp_dir / "LH_fs5.func.gii"
    RH_fs5_gii = temp_dir / "RH_fs5.func.gii"
    print(f"Saving fs5 GIFTIs...")
    save_func_gii(WL_fs5, LH_fs5_gii)
    save_func_gii(WR_fs5, RH_fs5_gii)
    
    # Resample to fsaverage6
    LH_fs6_gii = temp_dir / "LH_fs6.func.gii"
    RH_fs6_gii = temp_dir / "RH_fs6.func.gii"
    print(f"Resampling LH to fsaverage6...")
    resample_gii(LH_fs5_gii, LH_fs6_gii, sphereL_fs5, sphereL_fs6, areaL_fs5, areaL_fs6)
    print(f"Resampling RH to fsaverage5...")
    resample_gii(RH_fs5_gii, RH_fs6_gii, sphereR_fs5, sphereR_fs6, areaR_fs5, areaR_fs6)
    
    # Create output filenames
    stem = npz_path.stem  # e.g., "LEFT_sub-MSC01_0_seed_001_K_12_L_7..."
    out_LH = out_dir / f"LH_{stem}_fs6.func.gii"
    out_RH = out_dir / f"RH_{stem}_fs6.func.gii"
    
    # Copy to final output location
    import shutil
    shutil.copy(LH_fs6_gii, out_LH)
    shutil.copy(RH_fs6_gii, out_RH)
    
    print(f"\n✓ Saved:")
    print(f"  {out_LH}")
    print(f"  {out_RH}")
    
    # Load and verify
    LH_img = nib.load(out_LH)
    RH_img = nib.load(out_RH)
    print(f"\nVerification:")
    print(f"  LH: {len(LH_img.darrays)} columns, {LH_img.darrays[0].data.shape[0]} vertices")
    print(f"  RH: {len(RH_img.darrays)} columns, {RH_img.darrays[0].data.shape[0]} vertices")
    print(f"  Expected vertices per hemi: {nL_fs6}")
    print(f"  Expected columns (K*L): {K*L}")
    
    # Clean up temp
    shutil.rmtree(temp_dir)
    
    return out_LH, out_RH

if __name__ == "__main__":
    # Test files
    test_files = [	    
        "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC01/0/fs5_npz/LEFT_sub-MSC01_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.func_fs5_masked.npz"
    ]
    
    for npz_file in test_files:
        npz_path = Path(npz_file)
        if not npz_path.exists():
            print(f"WARNING: File not found: {npz_file}")
            continue
        
        # Create output directory in same location as input
        out_dir = npz_path.parent / "fs6_func_gii"
        out_dir.mkdir(exist_ok=True)
        
        try:
            process_npz(npz_path, out_dir)
        except Exception as e:
            print(f"ERROR processing {npz_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("All files processed!")


