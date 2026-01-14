#!/usr/bin/env python3
"""
upsample_npz_fs4_to_fs5.py
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

# Original fs4 mask
mask_fs4_path = "/scratch/groups/anishm/fmri/HCP_tSNR/renorm_H_tflip/mask_indices.npy"
mask_fs4_indices = np.load(mask_fs4_path).astype(int)
valid_fs4 = np.setdiff1d(np.arange(nL_fs4 + nR_fs4), mask_fs4_indices)
keepL_fs4 = valid_fs4[valid_fs4 < nL_fs4]
keepR_fs4 = valid_fs4[valid_fs4 >= nL_fs4] - nL_fs4

# Template paths
template_base = "/oak/stanford/groups/anishm/Adam/TemplateBrains/humans/surf/resample_fsaverage"
sphereL_fs4 = f"{template_base}/fsaverage4_std_sphere.L.3k_fsavg_L.surf.gii"
sphereR_fs4 = f"{template_base}/fsaverage4_std_sphere.R.3k_fsavg_R.surf.gii"
sphereL_fs5 = f"{template_base}/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii"
sphereR_fs5 = f"{template_base}/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii"
areaL_fs4 = f"{template_base}/fsaverage4.L.midthickness_va_avg.3k_fsavg_L.shape.gii"
areaR_fs4 = f"{template_base}/fsaverage4.R.midthickness_va_avg.3k_fsavg_R.shape.gii"
areaL_fs5 = f"{template_base}/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii"
areaR_fs5 = f"{template_base}/fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii"

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
    WLm = W[:len(keepL_fs4), :, :]
    WRm = W[len(keepL_fs4):, :, :]
    print(f"Split: LH={WLm.shape}, RH={WRm.shape}")
    
    # Expand to full fsaverage4 surfaces (with zeros for masked vertices)
    WL_fs4 = np.zeros((nL_fs4, K, L), np.float32)
    WR_fs4 = np.zeros((nR_fs4, K, L), np.float32)
    WL_fs4[keepL_fs4, :, :] = WLm
    WR_fs4[keepR_fs4, :, :] = WRm
    print(f"Expanded to fs4: LH={WL_fs4.shape}, RH={WR_fs4.shape}")
    
    # Save as fsaverage4 func.gii
    LH_fs4_gii = temp_dir / "LH_fs4.func.gii"
    RH_fs4_gii = temp_dir / "RH_fs4.func.gii"
    print(f"Saving fs4 GIFTIs...")
    save_func_gii(WL_fs4, LH_fs4_gii)
    save_func_gii(WR_fs4, RH_fs4_gii)
    
    # Resample to fsaverage5
    LH_fs5_gii = temp_dir / "LH_fs5.func.gii"
    RH_fs5_gii = temp_dir / "RH_fs5.func.gii"
    print(f"Resampling LH to fsaverage5...")
    resample_gii(LH_fs4_gii, LH_fs5_gii, sphereL_fs4, sphereL_fs5, areaL_fs4, areaL_fs5)
    print(f"Resampling RH to fsaverage5...")
    resample_gii(RH_fs4_gii, RH_fs5_gii, sphereR_fs4, sphereR_fs5, areaR_fs4, areaR_fs5)
    
    # Create output filenames
    stem = npz_path.stem  # e.g., "LEFT_sub-MSC01_0_seed_001_K_12_L_7..."
    out_LH = out_dir / f"LH_{stem}_fs5.func.gii"
    out_RH = out_dir / f"RH_{stem}_fs5.func.gii"
    
    # Copy to final output location
    import shutil
    shutil.copy(LH_fs5_gii, out_LH)
    shutil.copy(RH_fs5_gii, out_RH)
    
    print(f"\n✓ Saved:")
    print(f"  {out_LH}")
    print(f"  {out_RH}")
    
    # Load and verify
    LH_img = nib.load(out_LH)
    RH_img = nib.load(out_RH)
    print(f"\nVerification:")
    print(f"  LH: {len(LH_img.darrays)} columns, {LH_img.darrays[0].data.shape[0]} vertices")
    print(f"  RH: {len(RH_img.darrays)} columns, {RH_img.darrays[0].data.shape[0]} vertices")
    print(f"  Expected vertices per hemi: {nL_fs5}")
    print(f"  Expected columns (K*L): {K*L}")
    
    # Clean up temp
    shutil.rmtree(temp_dir)
    
    return out_LH, out_RH

if __name__ == "__main__":
    # Test files
    test_files = [ "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Smoothed6GroCon/sub-MSC06/0/LEFT_sub-MSC06_0_seed_000_K_8_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz" ]
    #    "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC01/0/LEFT_sub-MSC01_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz",
    #    "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC02/0/LEFT_sub-MSC02_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz",
    #    "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC03/0/LEFT_sub-MSC03_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz",
    #    "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC04/0/LEFT_sub-MSC04_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz",
    #    "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC05/0/LEFT_sub-MSC05_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz",
    #    "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC06/0/LEFT_sub-MSC06_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz",
    #    "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC07/0/LEFT_sub-MSC07_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz",
    #    "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC09/0/LEFT_sub-MSC09_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz",
    #    "/scratch/groups/anishm/fmri/MSC_individ_halfOfData_trainTest_Hnorm/sub-MSC10/0/LEFT_sub-MSC10_0_seed_001_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.01_L1H_0.0_all_Train.npz",
    #    "/oak/stanford/groups/anishm/fMRI_datasets/HCP-YA/Cortex_seed_003_K_12_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.001_L1H_0.0_all_halfprunedW.npz"
    #]
    
    for npz_file in test_files:
        npz_path = Path(npz_file)
        if not npz_path.exists():
            print(f"WARNING: File not found: {npz_file}")
            continue
        
        # Create output directory in same location as input
        out_dir = npz_path.parent / "fs5_func_gii"
        out_dir.mkdir(exist_ok=True)
        
        try:
            process_npz(npz_path, out_dir)
        except Exception as e:
            print(f"ERROR processing {npz_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("All files processed!")


