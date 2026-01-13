######
# Example usage: (please change all the directory paths to your own directory)
# python -i fit_fmri_jNMF_MSC_rest.py -K 10 -L 20 --lambda_OrthH 0.01 --lambda_OrthW 0.0 --lambda_L1W 0.0 --lambda_L1H 0.0 --lam 1e-5 --num_iters 10 --nH 9 --subj_st 1 --subj_en 11 --seed_st 0 --seed_en 1 --rescale01 --result_dir tmp --intermediate_dir tmp --data_dir /media//oak/stanford/groups/deissero/users/kwsheng/projects/JK_control_hub_paper/data/MSC_task/
######

import os
import numpy as np
import sys
sys.path.append('..')
from common_script.utils import rescale01, temporal_flip_concatenate, unit_flip_concatenate
from src.refit_jNMF import refit_jNMF
from fmri.util import get_upsamp_data_mask
from common_script.dataloader import AllSubjectsDataset
import argparse
from glob import glob
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-K', type=int, help='number of factors', default=25)
    parser.add_argument(
        '-L', type=int, help='factor duration (frames)', default=50)
    parser.add_argument(
        '--lambda_OrthH', type=float, 
        help='Penalize ||HSH^T||_1,i~=j; Encourages events-based factorizations',
        default=0.0)
    parser.add_argument(
        '--lambda_OrthW', type=float, 
        help='Penalize ||HSH^T||_1,i~=j; Encourages events-based factorizations',
        default=0.0)
    parser.add_argument(
        '--lambda_L1W', type=float, 
        help='Penalize ||HSH^T||_1,i~=j; Encourages events-based factorizations',
        default=0.0)
    parser.add_argument(
        '--lambda_L1H', type=float, 
        help='Penalize ||HSH^T||_1,i~=j; Encourages events-based factorizations',
        default=0.0)
    parser.add_argument(
        '--lam', type=float, 
        help='Penalize ||HSH^T||_1,i~=j; Encourages events-based factorizations',
        default=1e-5)
    parser.add_argument(
        '--tol', type=float, 
        help='Stopping tolerance',
        default=-np.inf)
    parser.add_argument(
        '--num_iters', type=int, 
        help='Maximum number of iterations for training',
        default=500)
    parser.add_argument(
        '--rescale01', help='rescale data to [0,1]', action='store_true')
    parser.add_argument(
        '--mean_subtraction', help='mean center the data', action='store_true')
    parser.add_argument(
        '--voxel_minmax', help='mean center the data', action='store_true')
    parser.add_argument(
        '--temporal_flip_concatenation', help='temporal flip and concatenate the data', action='store_true')
    parser.add_argument(
        '--unit_flip_concatenation', help='unit flip and concatenate the data', action='store_true')
    parser.add_argument(
        '--subj_st', type=int, 
        help='Start index of subjects',
        default=1)
    parser.add_argument(
        '--subj_en', type=int, 
        help='End index of subjects',
        default=11)
    
    parser.add_argument('--seed_st', type=int, required=True, help='the starting seed')
    parser.add_argument('--seed_en', type=int, required=True, help='the ending seed')

    parser.add_argument('--train_frac', type=float, required=True, help='training fraction')

    parser.add_argument('--up', type=float, required=True, help='upsampling ratio')
    
    parser.add_argument('--result_dir', type=str, required=True, help='result directory (e.g., /oak/stanford/groups/deissero/users/kwsheng/projects/JK_control_hub_paper/results/MSC_task/)')
    parser.add_argument('--data_dir', type=str, required=True, help='data directory (e.g., /oak/stanford/groups/deissero/users/kwsheng/projects/JK_control_hub_paper/data/MSC_task/)')
    parser.add_argument('--intermediate_dir', type=str, required=True, help='intermediate data directory')
    # adding argument for solution to use as initialization for personalization
    parser.add_argument('--initWs', type=str, required=True, help='the initial seqNMF solution you are initializing from')
    parser.add_argument('--subject', type=str, help='single subject like sub-MSC01')
    args = parser.parse_args()

    # Specify that we want our tensors on the GPU and in float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    # Helper function to convert between numpy arrays and tensors
    to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
    from_t = lambda tensor: tensor.to("cpu").detach().numpy()

    os.makedirs(args.result_dir, exist_ok = True)
    print(f'Running parameters - K: {args.K}, L: {args.L}, lambda_OrthH: {args.lambda_OrthH}, lambda_OrthW: {args.lambda_OrthW}, lambda_L1W: {args.lambda_L1W}, lambda_L1H: {args.lambda_L1H}, lam: {args.lam}, num_iters: {args.num_iters}, subj_st: {args.subj_st}, subj_en: {args.subj_en}, seed_st: {args.seed_st}, seed_en: {args.seed_en}, rescale01: {args.rescale01}, mean_subtraction: {args.mean_subtraction}, voxel_minmax: {args.voxel_minmax}, temporal_flip_concatenation: {args.temporal_flip_concatenation}, unit_flip_concatenation: {args.unit_flip_concatenation}, train_frac: {args.train_frac}, up: {args.up}, result_dir: {args.result_dir}, intermediate_dir: {args.intermediate_dir}, data_dir: {args.data_dir}, initial_Ws: {args.initWs}', flush=True)
    
    ### 
    # if you haven't combine all datasets before, please run the following
    # otherwise, you can skip this part
    up = args.up # originally I used 10
    TR = 1.2 #s
    ntrunc = 1
    # loading in standard spatial mask used for initial fit on HCP data
    mask_idx = np.load('/oak/stanford/groups/anishm/fMRI_datasets/HCP-YA/TSNR_Mask_Indices.py').astype(int)
    n_datasets = 0
    data_prefix = 'faln_xr3d_uwrp_on_ln_MNI152_T1_2mm_norm_bpss_resid_uncr.resamp_to_3k_fsaverage4.on_MNI_152_smooth2p25.dtseries'
    existing_files = glob(os.path.join(args.data_dir, '**', f'*{data_prefix}.nii'), recursive=True)
    subject_list = list()
    # set structures to load in
    structures=['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT']
    # make subject list 
    #if args.subject:
    #    subject_ids = [int(args.subject.split('sub-MSC')[-1])]
    #else:
    #    subject_ids = list(range(args.subj_st, args.subj_en))
    subject_ids = [args.subject]
    for subject_id in subject_ids:
        args.subject = f'{subject_id}'
        # FIXME: the dude's data looks weird, so skip it...
        if subject_id == 8:
            continue
	# set subj dir
        subj_dir = f'{args.data_dir}/{args.subject}/'
        # set subject tag
        #subject_tag = f"{args.subject}_{args.parity}"   # e.g., sub-MSC01_odd
        x_path = f"{args.intermediate_dir}/{args.subject}_X.npy"
        m_path = f"{args.intermediate_dir}/{args.subject}_M.npy"
	# initialize X and M
        X = []
        M = []
        # does it exist? check
        #parity_ses_ids = [s for s in range(1, 11)
        #    if (args.parity == 'odd'  and s % 2 == 1)
        #    or (args.parity == 'even' and s % 2 == 0)]

        # count extant npys fitting the bill
        if os.path.exists(x_path) and os.path.exists(m_path):
            print(f"{args.subject}: intermediates already present; skipping rebuild.")
            subject_list = [args.subject]
            n_datasets   = 1
            continue
        
        #fname_fmri = f'{subj_dir}/ses-func{ses_id:02d}/cifti/sub-MSC{subject_id:02d}_ses-func{ses_id:02d}_task-rest_bold_32k_fsLR_fsavg4.dtseries.nii'
        fname_fmri = f'{subj_dir}/{args.subject}_faln_xr3d_uwrp_on_ln_MNI152_T1_2mm_norm_bpss_resid_uncr.resamp_to_3k_fsaverage4.on_MNI_152_smooth3p00.dtseries.nii'
        #fname_mask = f'{subj_dir}/ses-func{ses_id:02d}/cifti/sub-MSC{subject_id:02d}_ses-func{ses_id:02d}_task-rest_bold_32k_fsLR_tmask.txt'
        fname_mask = f'{subj_dir}/{args.subject}_faln_xr3d_uwrp_on_ln_MNI152_T1_2mm_norm_tmask.txt'

        print(fname_fmri, fname_mask)

        try:
            up_data_ctx, t_mask, _ = get_upsamp_data_mask(
                fname_fmri,
                fname_mask,
                structures=structures,
                TR=TR,
                up=up,
                ntrunc=ntrunc
            )
        except FileNotFoundError:
            continue
	# actually upsample the data, use spatial mask
        up_data_ctx = np.delete(up_data_ctx, mask_idx, axis=0)
        N, T = up_data_ctx.shape
	#dummy mask because data is already masked from mask_idx
        spatial_mask = np.ones((N, 1), dtype=bool)
        temporal_mask = np.tile(t_mask, (N, 1))
        mask = spatial_mask & temporal_mask
	# zero-padding now within this script
        up_data_ctx = np.pad(up_data_ctx, ((0,0), (args.L,args.L))).astype(np.float32)
	# same padding to "mask"
        mask = np.pad(mask, ((0,0), (args.L,args.L))).astype(np.bool_)
        X.append(up_data_ctx)
        M.append(~mask)

        # concatenate all the sessions together
        X = np.concatenate(X, axis=1)
        M = np.concatenate(M, axis=1)
        # apply concat of choice
        if args.temporal_flip_concatenation:
            up_data_ctx, M = temporal_flip_concatenate(X, M)
            # divide by time series max for 0-1 scaling
            up_data_ctx_max = up_data_ctx.max()
            X = up_data_ctx / up_data_ctx_max
        if args.unit_flip_concatenation:
            up_data_ctx, M = unit_flip_concatenate(X, M)
            # same as above: 0-1 scaling via within-scan max denominator
            up_data_ctx_max = up_data_ctx.max()
            X = up_data_ctx / up_data_ctx_max 

        #for i_X in range(len(X)):
        #    train_len = int(X[i_X].shape[1] * args.train_frac)
        #    print(f'sub-MSC{subject_id:02} ses-{i_X+1:02d} time series length: {X[i_X].shape[1]}, training length: {train_len}', flush=True)
        #    X[i_X] = X[i_X][:, :train_len]
        #    M[i_X] = M[i_X][:, :train_len]
        np.save(x_path, X.astype(np.float32))
        np.save(m_path, M.astype(np.bool_))
        subject_list = subject_ids
        n_datasets   = len(subject_list)
        # np.save(f'{args.intermediate_dir}/{args.subject}_X_{n_datasets}.npy', X)
        # np.save(f'{args.intermediate_dir}/{args.subject}_M_{n_datasets}.npy', M)
        # subject_list.append(args.subject)
        # n_datasets += 1
    # assert n_datasets == args.nH, f'number of datasets {n_datasets} != nH {args.nH}'
    args.nH = n_datasets
    ###
    

    for seed in range(args.seed_st, args.seed_en):
        data_loader = DataLoader(
            AllSubjectsDataset(args.intermediate_dir, n_datasets, subject_list, args.L, args.K, seed),
            batch_size=1,
            shuffle=True,
            num_workers=1,
        )
        result_file = (
            f'{args.result_dir}/LEFT_{args.subject}_'
            f'seed_{seed:03}_K_{args.K}_L_{args.L}_lam_e{np.log10(args.lam)}'
            f'_orthH_{args.lambda_OrthH}_orthW_{args.lambda_OrthW}'
            f'_L1W_{args.lambda_L1W}_L1H_{args.lambda_L1H}_all.npz'
        ) 
        refit_jNMF(
            data_loader=data_loader,
            result_file=result_file,
            K=args.K, L=args.L, lam=args.lam, lambda_OrthH=args.lambda_OrthH, lambda_OrthW=args.lambda_OrthW,
            lambda_L1W=args.lambda_L1W, lambda_L1H=args.lambda_L1H, nH=args.nH, seed=seed, num_iters=args.num_iters,
            device=device, dtype=dtype, template_npz=args.initWs, renorm_type='renorm_H_norm',
            no_improve_tol=1500,
        )
