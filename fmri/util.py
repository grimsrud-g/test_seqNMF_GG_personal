import nibabel as nib
import numpy as np
import scipy as sp

store_dir = '../'
SURF_DIR = f'{store_dir}/data/4k_surfaces/downsampled_4k_surfs/'
MOTION_DIR = f'{store_dir}/data/motion'

def get_upsamp_data(subject, session, up=10, ntrunc=1, TR = 2.2, 
        structures=['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT'],
        procopt = 'gsr'):

    fmri_dir = f'{store_dir}/data/MSC_{procopt}/{subject}/cifti_timeseries_normalwall_atlas_4k/'
    fname_fmri = f'{fmri_dir}/CIFTI2-{session}_LR_surf_4k_fs_LR_smooth5.1_subcort.dtseries.nii'
    fname_mot = f'{MOTION_DIR}/{session}_faln_dbnd_xr3d.FD'

    # Get the file, data, and header.
    img = nib.load(fname_fmri)
    data = img.get_fdata()
    hdr = img.header

    axs = [hdr.get_axis(i) for i in range(img.ndim)]
    rois = list(axs[1].iter_structures())

    idx_dict = dict((x[0], x[1]) for x in rois)
    idx_ctx = np.r_[tuple([idx_dict[s] for s in structures])]
    data_ctx = data[ntrunc:, idx_ctx] #NB truncating the first frame
    T, N = data_ctx.shape

    frame_times = np.arange(T)*TR
    up_data_ctx, up_frame_times = sp.signal.resample(data_ctx, up*T, t=frame_times)
    fd = np.loadtxt(fname_mot)[ntrunc:, 0]
    
    up_fd = sp.signal.resample(fd, up*T)
    
    return up_data_ctx.T, up_fd, up_frame_times

def get_upsamp_data_mask(fname_fmri, fname_mask, up=10, ntrunc=1, TR = 2.2, 
        structures=['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT']):

    # Get the file, data, and header.
    img = nib.load(fname_fmri)
    data = img.get_fdata()
    hdr = img.header

    axs = [hdr.get_axis(i) for i in range(img.ndim)]
    rois = list(axs[1].iter_structures())

    idx_dict = dict((x[0], x[1]) for x in rois)
    data_ctx = np.concatenate([data[ntrunc:, idx_dict[s]] for s in structures], axis=1) #NB truncating the first frame
    T, N = data_ctx.shape

    frame_times = np.arange(T)*TR

    if isinstance(fname_mask, str):
        mask_raw = np.loadtxt(fname_mask)
    elif isinstance(fname_mask, np.ndarray):
        mask_raw = fname_mask
    else:
        mask_raw = None

    if mask_raw is None:
        valid_mask = np.ones(T, dtype=np.bool_)
    else:
        mask_arr = np.asarray(mask_raw)
        mask_arr = np.squeeze(mask_arr)
        mask_arr = mask_arr[ntrunc:]
        if mask_arr.shape[0] < T:
            raise ValueError("Mask shorter than fMRI time series after truncation")
        mask_arr = mask_arr[:T]
        valid_mask = mask_arr.astype(np.bool_)

    if not np.any(valid_mask):
        raise ValueError("No unmasked time points available for resampling")

    data_valid = data_ctx[valid_mask]
    times_valid = frame_times[valid_mask]

    up_samples = max(1, np.floor((times_valid[-1] - times_valid[0]) * up / TR + 0.5).astype(np.int32) + 1)
    up_data_ctx, up_frame_times = sp.signal.resample(data_valid, up_samples, t=times_valid, axis=0)

    up_data_ctx = up_data_ctx.T
    t_mask = np.ones(up_data_ctx.shape[1], dtype=np.bool_)

    return up_data_ctx, t_mask[None], up_frame_times
