#!/bin/bash
#SBATCH --job-name=baeBrains
#SBATCH --output=/oak/stanford/groups/anishm/grimsrud/test_seqNMF/code/output_logs/Dev_Pers_seed000."%j".o
#SBATCH --error=/oak/stanford/groups/anishm/grimsrud/test_seqNMF/code/output_logs/Dev_Pers_seed000."%j".e
#SBATCH -p owners,normal,anishm
#SBATCH --mem=70GB
#SBATCH -t 9:00:00
ml python/3.12
source ~/.bashrc
source /oak/stanford/groups/anishm/grimsrud/envs/seqNMFenv/bin/activate 
python3 refit_fmri_jNMF_Babies_rest.py -K 8 -L 7 --lambda_OrthH 0.01 --lambda_OrthW 0.001 --lambda_L1W 0.01 --lambda_L1H 0.0 --lam 1e-05 --num_iters 500 --subj_st 1 --subj_en 1 --seed_st 0 --seed_en 1 --rescale01 --result_dir /scratch/groups/anishm/grimsrud/test_seqNMF/Dev_Pers_Results_K_8 --intermediate_dir /scratch/groups/anishm/grimsrud/test_seqNMF/Dev_Pers_Intermed_K_8 --data_dir /oak/stanford/groups/anishm/fMRI_datasets/PatKuhl_development --temporal_flip_concatenation --train_frac 1 --up 1.667 --subject ${1} --initWs /oak/stanford/groups/anishm/grimsrud/test_seqNMF/HCPFinalTestFinal_K_8_all/Cortex_seed_000_K_8_L_7_lam_e-5.0_orthH_0.01_orthW_0.001_L1W_0.001_L1H_0.0_all_smoothed6_halfprunedW.npz
