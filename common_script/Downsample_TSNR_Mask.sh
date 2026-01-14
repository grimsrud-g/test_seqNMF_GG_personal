#!/bin/bash
# downsample_tsnr_fslr_to_fs5.sh

# Input file
InputDscalar=/scratch/groups/anishm/fmri/HCP_PreProcd_group_average_tsnr.dscalar.nii

# Output directory
OutDir=/scratch/groups/anishm/fmri/HCP_tSNR
mkdir -p ${OutDir}

# Temp directory
TempDir=${OutDir}/temp_fs5
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
    ${TemplatePath}/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii \
    ADAP_BARY_AREA ${TempDir}/tsnr_L_fs5.func.gii \
    -area-metrics ${TemplatePath}/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii \
    ${TemplatePath}/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii

# Right hemisphere: 32k fsLR -> fsaverage5
wb_command -metric-resample ${TempDir}/tsnr_R_32k.func.gii \
    ${TemplatePath}/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii \
    ${TemplatePath}/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii \
    ADAP_BARY_AREA ${TempDir}/tsnr_R_fs5.func.gii \
    -area-metrics ${TemplatePath}/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii \
    ${TemplatePath}/fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii

echo "Creating fsaverage5 dscalar..."
# Combine into new dscalar
OutputDscalar=${OutDir}/HCP_PreProcd_group_average_tsnr_fs5.dscalar.nii
wb_command -cifti-create-dense-scalar ${OutputDscalar} \
    -left-metric ${TempDir}/tsnr_L_fs5.func.gii \
    -right-metric ${TempDir}/tsnr_R_fs5.func.gii

echo "Done! Output saved to: ${OutputDscalar}"

# Optional: keep the individual hemisphere files too
cp ${TempDir}/tsnr_L_fs5.func.gii ${OutDir}/tsnr_L_fs5.func.gii
cp ${TempDir}/tsnr_R_fs5.func.gii ${OutDir}/tsnr_R_fs5.func.gii

# Clean up temp
rm -rf ${TempDir}
