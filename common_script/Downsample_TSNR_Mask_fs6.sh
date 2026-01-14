#!/bin/bash
# downsample_tsnr_fslr_to_fs5.sh
ml biology
ml workbench
ml freesurfer

# Input file
InputDscalar=/oak/stanford/groups/anishm/fMRI_datasets/HCP_PreProcd_group_average_tsnr.dscalar.nii

# Output directory
OutDir=/scratch/groups/anishm/fmri/HCP_tSNR
mkdir -p ${OutDir}

# Temp directory
TempDir=${OutDir}/temp_fs6
mkdir -p ${TempDir}

# Template paths
TemplatePath=/oak/stanford/groups/anishm/Adam/TemplateBrains/humans/surf/resample_fsaverage

echo "Separating hemispheres from fsLR dscalar..."
# Separate hemispheres from 32k fsLR
wb_command -cifti-separate ${InputDscalar} COLUMN \
    -metric CORTEX_LEFT ${TempDir}/tsnr_L_32k.func.gii

wb_command -cifti-separate ${InputDscalar} COLUMN \
    -metric CORTEX_RIGHT ${TempDir}/tsnr_R_32k.func.gii

echo "Resampling to fsaverage5..."
# Left hemisphere: 32k fsLR -> fsaverage5
wb_command -metric-resample ${TempDir}/tsnr_L_32k.func.gii \
    ${TemplatePath}/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii \
    ${TemplatePath}/fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii \
    ADAP_BARY_AREA ${TempDir}/tsnr_L_fs6.func.gii \
    -area-metrics ${TemplatePath}/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii \
    ${TemplatePath}/fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii

# Right hemisphere: 32k fsLR -> fsaverage5
wb_command -metric-resample ${TempDir}/tsnr_R_32k.func.gii \
    ${TemplatePath}/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii \
    ${TemplatePath}/fsaverage6_std_sphere.R.41k_fsavg_R.surf.gii \
    ADAP_BARY_AREA ${TempDir}/tsnr_R_fs6.func.gii \
    -area-metrics ${TemplatePath}/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii \
    ${TemplatePath}/fsaverage6.R.midthickness_va_avg.41k_fsavg_R.shape.gii

echo "Creating fsaverage5 dscalar..."
# Combine into new dscalar
OutputDscalar=${OutDir}/HCP_PreProcd_group_average_tsnr_fs6.dscalar.nii
wb_command -cifti-create-dense-scalar ${OutputDscalar} \
    -left-metric ${TempDir}/tsnr_L_fs6.func.gii \
    -right-metric ${TempDir}/tsnr_R_fs6.func.gii

echo "Done! Output saved to: ${OutputDscalar}"

# Optional: keep the individual hemisphere files too
cp ${TempDir}/tsnr_L_fs6.func.gii ${OutDir}/tsnr_L_fs6.func.gii
cp ${TempDir}/tsnr_R_fs6.func.gii ${OutDir}/tsnr_R_fs6.func.gii

# Clean up temp
rm -rf ${TempDir}
