import glob
import numpy as np
import nibabel as nib
from pathlib import Path
import sys

nL = nR = 2562
mask_both = "/scratch/groups/anishm/fmri/HCP_tSNR/renorm_H_tflip/mask_indices.npy"
# mask_both = "/scratch/groups/anishm/fmri/BRAINS_Test50subj600Epoch/tempflip/mask_indices.npy"
# mask_both = "/oak/stanford/groups/anishm/fMRI_datasets/HCP-YA/TSNR_Mask_Indices.npy"
mask_both = "/scratch/groups/anishm/fmri/HCPFinalTestFinal/mask_indices.npy"

valid = np.setdiff1d(np.arange(nL + nR), np.load(mask_both).astype(int))
keepL = valid[valid < nL]
keepR = valid[valid >= nL] - nL  # shift to RH local index

def save_func_gii(arr3d, out_path):
    n, K, L = arr3d.shape
    arr2d = arr3d.reshape(n, K * L, order="C")
    darrays = [nib.gifti.GiftiDataArray(arr2d[:, i].astype(np.float32))
               for i in range(arr2d.shape[1])]
    nib.save(nib.gifti.GiftiImage(darrays=darrays), out_path)

# loop through all seed files
indir = Path(sys.argv[1])
# indir = Path("/scratch/groups/anishm/fmri/HCPFinalTest_Final60_2")
files = sorted(glob.glob(str(indir / "*seed*.npz")))

for f in files:
    fname = Path(f).name
    npz = np.load(f)
    W = npz["W"].astype(np.float32)        # shape: (len(valid), K, L)
    K, L = W.shape[1], W.shape[2]

    # split rows: first len(keepL) -> LH, remainder -> RH
    WLm = W[:len(keepL), :, :]
    WRm = W[len(keepL):, :, :]

    WL = np.zeros((nL, K, L), np.float32); WL[keepL, :, :] = WLm
    WR = np.zeros((nR, K, L), np.float32); WR[keepR, :, :] = WRm

    # build output names using the input stem
    stem = fname.replace(".npz", "")
    outL = indir / f"LH_{stem}_W_timeSeries.func.gii"
    outR = indir / f"RH_{stem}_W_timeSeries.func.gii"

    save_func_gii(WL, outL)
    save_func_gii(WR, outR)
    print(f"{fname}: wrote {outL.name}, {outR.name}")

