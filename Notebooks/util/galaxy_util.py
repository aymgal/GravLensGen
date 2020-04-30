import astropy.io.fits as pyfits
import numpy as np
import scipy

def get_galsim_image(image_size, pixel_size, catalog_dir=None, catalog_name=None, catalog_index=0, 
                     galsim_scale=1, galsim_angle=0, galsim_shear_eta=0, galsim_shear_beta=0,
                     galsim_center_x=0, galsim_center_y=0,
                     psf_size=49, psf_pixel_size=0.074, galaxy_type='real',
                     psf_type='real', psf_gaussian_fwhm=0.2, no_convolution=False,
                     draw_image_method='auto', verbose=False, cosmos_exclusion_level='marginal',
                     cosmos_min_hlr=0, cosmos_max_hlr=0, cosmos_min_flux=0, cosmos_max_flux=0,
                     cut_negative_flux=False):
    """
    Generates a realistic galaxy using galsim (HST F814W extracted).
    """
    import galsim

    # Load catalog of galaxies
    cat = galsim.COSMOSCatalog(dir=catalog_dir, file_name=catalog_name, preload=False,
                               exclusion_level=cosmos_exclusion_level,
                               min_hlr=cosmos_min_hlr, max_hlr=cosmos_max_hlr, 
                               min_flux=cosmos_min_flux, max_flux=cosmos_max_flux)
    if verbose:
        print("Number of galaxies in catalog '{}' : {}".format(catalog_name, cat.nobjects))
    
    # Get galaxy object
    gal = cat.makeGalaxy(catalog_index, gal_type=galaxy_type, noise_pad_size=0)
    
    # effective pixel size
    pixel_size_eff = pixel_size / galsim_scale
    
    # apply rotation -> we do it after accessing the PSF otherwise raises an error
    #if angle != 0:
    #    gal_rot = gal.rotate(angle * galsim.radians)
    
    if psf_type == 'real':
        if galaxy_type == 'real':
            # Get original (untouched) PSF
            psf_kernel_untouched = gal.psf_image.array
            # Dilate the PSF to match required resolution
            psf_dilate_factor = psf_pixel_size / 0.074  # taken for HST F814W band
            psf = gal.original_psf.dilate(psf_dilate_factor).withFlux(1.)
            # Get the actual image of the psf
            # note that we set 'use_true_center' to False to make sure that the PSF is centered on a pixel (event if even-size image)
            psf_kernel = psf.drawImage(nx=psf_size, ny=psf_size, use_true_center=False, 
                                       scale=pixel_size_eff).array
        else:
            psf_kernel = np.zeros((image_size, image_size))
            psf_kernel_untouched = np.zeros((image_size, image_size))
    elif psf_type == 'gaussian':
        # Dilate the PSF to match required resolution
        psf = galsim.Gaussian(fwhm=psf_gaussian_fwhm, flux=1.0)
        # Get the actual image of the psf
        # note that we set 'use_true_center' to False to make sure that the PSF is centered on a pixel (event if even-size image)
        psf_kernel = psf.drawImage(nx=psf_size, ny=psf_size, use_true_center=False, 
                                   scale=pixel_size_eff).array
        psf_kernel_untouched = np.zeros((image_size, image_size))
    
    # apply rotation
    if galsim_angle != 0:
        gal = gal.rotate(galsim_angle * galsim.radians)

    if galsim_shear_eta != 0 or galsim_shear_beta != 0:
        shear = galsim.Shear(eta=galsim_shear_eta, beta=galsim_shear_beta * galsim.radians)
        gal = gal.shear(shear)
    
    # Performs convolution with PSF
    if no_convolution is False:
        if psf_type == 'real' and galaxy_type == 'parametric':
            print("WARNING : no 'real' PSF convolution possible with gal_type 'parametric' !")
        else:
            gal = galsim.Convolve(gal, psf)
    
    # Project galaxy on an image grid
    image_galaxy = gal.drawImage(nx=image_size, ny=image_size, use_true_center=True,
                                 offset=[galsim_center_x, galsim_center_y],
                                 scale=pixel_size_eff, method=draw_image_method).array
    
    if cut_negative_flux:
        image_galaxy[image_galaxy < 0] = 0.
    
    return image_galaxy, psf_kernel, psf_kernel_untouched


def get_spiral_galaxy(galaxy_name, num_pix, scale, image_dir=None, sigma_gaussian_conv=5):
    if image_dir is None:
        image_dir = os.path.join('Data', 'TDLMC_sources')
    image_path = os.path.join(image_dir, '{}_fix.fits'.format(galaxy_name))
    # read data
    with pyfits.open(image_path) as f:
        image_raw = f[0].data
    
    # we slightly convolve the image with a Gaussian convolution kernel
    image_conv = scipy.ndimage.filters.gaussian_filter(image_raw, sigma_gaussian_conv, 
                                                       mode='nearest', truncate=6)

    return image_conv
