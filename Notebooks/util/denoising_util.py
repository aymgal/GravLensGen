import numpy as np
from lenstronomy.LightModel.Profiles.starlets import Starlets
from slitronomy.Util.util import hard_threshold


def starlets_denoising(raw_image, threshold_level, n_scales=4, scales_bool_list=None):
    starlets_profile = Starlets()
    raw_coeffs = starlets_profile.decomposition_2d(raw_image, n_scales=n_scales)
    #print(raw_coeffs.shape, np.mean(raw_coeffs), np.median(np.abs(raw_coeffs-np.median(raw_coeffs))))

    # filter out some coefficients in low frequency scales
    cleaned_coeffs = np.copy(raw_coeffs)
    if scales_bool_list is None:
        # by default we threshold only first scales
        scales_bool_list = [True] + [False]*(n_scales-1)
    for s in range(n_scales-1):  # we ignore largest frequencies (coarsest scale)
        if scales_bool_list[s] is True:
            cleaned_coeffs[s, :, :] = hard_threshold(cleaned_coeffs[s, :, :], thresh=threshold_level)
    
    # reconstruct the image
    cleaned_image = starlets_profile.function_2d(cleaned_coeffs, n_scales=n_scales, n_pixels=raw_image.size)
    return cleaned_image