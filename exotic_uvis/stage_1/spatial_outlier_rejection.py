from tqdm import tqdm

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
from astropy.stats import sigma_clip

from exotic_uvis.plotting import plot_exposure
from exotic_uvis.stage_2 import spatial_profile_smooth


def spatial_smoothing(obs, type='1D_smooth', kernel=11, sigma=10, bounds_set=[[260, 370, 640, 1100],],
                      verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Uses 2D median filtering to catch and remove hot pixels.

    Args:
        obs (xarray): obs.images contains the dataset we are cleaning.
        type (str): options are "1D_smooth" (to use row-wise scipy
        median-filtering),  "2D_smooth" (to use scipy 2D median-filtering),
        and "polyfit" (to fit row-wise polynomials).
        kernel (int or tup, optional): the size of the kernel used to compute
        the median-filtered image. If using 1D_smooth, should be an odd int. If
        using 2D_smooth, should be a tuple of two odd ints. Defaults to 11.
        sigma (float, optional): threshold at which to remove an outlier.
        Defaults to 10.
        bounds_set (array-like, optional): whether to only perform spatial
        smoothing on a subset of the array, for time-saving. If None, corrects
        for the full frame. Defaults to [[260, 370, 640, 1100],].
        verbose (int, optional): how detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show.
        Defaults to 0.
        save_plots (int, optional): how many plots you want to save.
        Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        xarray: obs with images cleaned and data quality flags updated.
    """

    # get images and initialize
    images = obs.images.data
    all_xhits, all_yhits = [], []

    # generate random exposure number for plotting
    plot_ind = np.random.randint(0, np.shape(images)[0])

    # check if bound were given
    if bounds_set is None:
        bounds_set = [[0, -1, 0, -1]]

    # iterate over all images
    for i, image in enumerate(tqdm(images, desc = 'Computing spatial profile... Progress:'),
                              disable=(verbose==0)):
        # iterate over all bounds
        for bounds in bounds_set:
            # define sub_image according to given bounds
            sub_image = image[bounds[0]:bounds[1], bounds[2]:bounds[3]].copy()
           
            # remove outliers with 1D smoothing in the dispersion direction
            if type == '1D_smooth':
                Pprof, sub_image_clean, xhits, yhits = spatial_profile_smooth(sub_image,
                                                                              threshold = sigma,
                                                                              kernel = kernel,
                                                                              show_plots=show_plots,
                                                                              save_plots=save_plots)           
            # remove outliers with 2D smoothing 
            elif type == '2D_smooth':
                Pprof, sub_image_clean, xhits, yhits = spatial_2D_smooth(sub_image,
                                                                         kernel=kernel,
                                                                         sigma=sigma)

            # remove outliers with other routines
            elif type == 'polyfit':
                print('More cleaning options')

            # save position of corrected outliers
            all_xhits = np.concatenate((all_xhits, xhits))
            all_yhits = np.concatenate((all_yhits, yhits))

            # if true, plot one exposure and draw location of all detected cosmic rays in that exposures
            if save_plots > 0 or show_plots > 0:
                if (show_plots > 0 or save_plots > 0) and i == plot_ind:
                    plot_exposure([sub_image, sub_image_clean], min = 1e0, 
                                title = f'Spatial Bad Pixel removal Exposure {i}', 
                                show_plot=(show_plots > 0), save_plot=(save_plots > 0),
                                output_dir=output_dir,
                                filename = [f'spatialsmooth_before_correction_frame{i}', f'spatialsmooth_after_correction_frame{i}'])

                    plot_exposure([sub_image], scatter_data=[xhits, yhits], min = 1e0, 
                                title = f'Location of corrected pixels for Exposure {i}', mark_size = 1,
                                show_plot=(show_plots > 0), save_plot=(save_plots > 0),
                                output_dir=output_dir, filename = [f'spatialsmooth_location_frame{i}'])
                
                elif (show_plots == 2 or save_plots == 2):
                    plot_exposure([sub_image, sub_image_clean], min = 1e0, 
                                title = f'Spatial Bad Pixel removal Exposure {i}', 
                                show_plot=(show_plots == 2), save_plot=(save_plots == 2),
                                output_dir=output_dir,
                                filename = [f'spatialsmooth_before_correction_frame{i}', f'spatialsmooth_after_correction_frame{i}'])

                    plot_exposure([sub_image], scatter_data=[xhits, yhits], min = 1e0, 
                                title = f'Location of corrected pixels for Exposure {i}', mark_size = 1,
                                show_plot=(show_plots == 2), save_plot=(save_plots == 2),
                                output_dir=output_dir, filename = [f'spatialsmooth_location_frame{i}'])
            
            # update image
            image[bounds[0]:bounds[1], bounds[2]:bounds[3]] = sub_image_clean
          
    return obs


def spatial_2D_smooth(sub_image, kernel=(5,5), sigma=10):
    """Uses scipy.signal.medfil2d to correct spatial outliers.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        sub_image (array-like): a frame to clean of outliers.
        kernel (tuple, optional): tuple of two odd ints which define the
        kernel used for smoothing. Defaults to (5,5).
        sigma (float, optional): threshold at which to remove an outlier.
        Defaults to 10.

    Returns:
        array-like, array-like, array-like, array-like: the median-filtered
        image, cleaned image, and maps of where pixels were hit in x and y.
    """

    # build the median-filtered model
    Pprof = medfilt2d(sub_image,kernel_size=kernel)

    # look for outliers
    difference_image = sub_image - Pprof
    clipped = sigma(difference_image,sigma=sigma,axis=0)
    mask = clipped.mask
    int_mask = mask.astype(float) * Pprof
    test = (~mask).astype(float)

    # replace outliers in the data
    sub_image_clean =  (sub_image*test) + int_mask

    # identify where changes were made
    bad_pix = np.argwhere(sub_image!=sub_image_clean)
    xhits = np.array([i[0] for i in bad_pix])
    yhits = np.array([i[1] for i in bad_pix])

    return Pprof, sub_image_clean, xhits, yhits


def laplacian_edge_detection(obs, sigma=10, factor=2, n=2, build_fine_structure=False, contrast_factor=5,
                             verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Uses Laplacian Edge Detection (van Dokkum 2001) to detect cosmic rays
    and hot/cold pixels.

    Args:
        obs (xarray): obs.images DataSet contains the images.
        sigma (float, optional): sigma to use for detecting bad pixels and
        replacing them. Defaults to 10.
        factor (int, optional): subsampling factor, minimum value 2 to work.
        Higher values increase computation time but don't tend to improve
        the routine much, so best left at 2. Defaults to 2.
        n (int, optional): how many iterations you want to run. Useful for
        catching large blobs of bad pixels, as LED detects edges and not
        interiors. Defaults to 2.
        build_fine_structure (bool, optional): whether to build a fine structure
        model to protect the trace against LED. Defaults to False.
        contrast_factor (int, optional): the threshold for rejection when a fine
        structure model is in use. Defaults to 5.
        verbose (int, optional): how detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show.
        Defaults to 0.
        save_plots (int, optional): how many plots you want to save.
        Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        xarray: obs with the .data cleaned of bad pixels and the .data_quality
        updated to reflect where bad pixels were found.
    """

    # Define the Laplacian kernel.
    l = 0.25*np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

    # Iterate over each frame one at a time until the iteration stop condition is met by each frame.
    if verbose >= 1:
        print("Cleaning threshold=%.1f outliers with Laplacian edge detection..." % sigma)
    for k in tqdm(range(obs.images.shape[0]),
                  desc="Applying LED... Progress:",
                  disable=(verbose < 2)):
        # Get the frame, errors, and dq array as np.array objects so we can operate on them.
        data_frame = obs.images[k].values
        errs = obs.errors[k].values
        dq = obs.data_quality[k].values

        # Track outliers flagged in this frame and iterations performed.
        bad_pix_removed = 0
        iteration_N = 1

        # Plots of data before LED is applied.
        if (show_plots == 1 or save_plots == 1) and k == 0:
            plot_exposure([obs.images.data[0]],
                          show_plot=(show_plots>=1), save_plot=(save_plots>=1), 
                          output_dir=output_dir, filename = ['LED_before_correction_0'])
        
        elif show_plots == 2 or save_plots == 2:
            plot_exposure([obs.images.data[k]],
                          show_plot=(show_plots==2), save_plot=(save_plots==2), 
                          output_dir=output_dir, filename = ['LED_before_correction_{}'.format(k)])

        # Then start iterating over this frame and keep going until the iteration stop condition is met.
        stop_iterating = False
        while not stop_iterating:
            # Estimate readnoise value.
            var2 = errs**2 - data_frame
            var2[var2 < 0] = 0 # enforce positivity.
            rn = np.sqrt(var2) # estimate readnoise array

            # Build the noise model.
            noise_model = build_noise_model(data_frame, rn)
            if build_fine_structure:
                F = build_fine_structure_model(data_frame)

            # Subsample the array.
            subsample, original_shape = subsample_frame(data_frame, factor=factor)
            
            # Convolve subsample with laplacian.
            lap_img = np.convolve(l.flatten(),subsample.flatten(),mode='same').reshape(subsample.shape)
            lap_img[lap_img < 0] = 0 # force positivity
            
            # Resample laplacian-convolved subsampled image to original size.
            resample = resample_frame(lap_img, original_shape)

            # Divide by the noise model scaled by the resampling factor.
            S = resample/(factor*noise_model)
            
            # Remove sampling flux to protect data from being targeted by LED.
            S = S - median_filter(S, size=5)

            # Spot outliers.
            S[np.abs(S) < sigma] = 0 # any not zero after this are rays.
            S[S!=0] = 1 # for visualization and comparison to fine structure model.

            # If we have a fine structure model, we also need to check the contrast.
            if build_fine_structure:
                contrast_image = resample/F
                contrast_image[contrast_image < contrast_factor] = 0 # any not zero after this are rays.
                contrast_image[contrast_image!=0] = 1 # for visualization and comparison to sampling flux model.
                
                # Then we need to merge the results of S = Laplacian_image/factor*noise_model - sampling_flux
                # and contrast_image = Laplacian_image/Fine_structure_model so that we only take where both are 1.
                S = np.where(S == contrast_image, S, 0)

            # Ignore the 0th order, it's a dead end of endless masking.
            xmid = int(S.shape[1]/2)
            S[0:-1,xmid-70:xmid+70] = 0 # FIX: currently hardcoded to assume the source / 0th order is near the middle of the frame.

            # Report where data quality flags should be added and count pixels to be replaced.
            dq = np.where(S != 0, 1, dq)
            bad_pix_last_frame = -100
            if iteration_N != 1:
                bad_pix_last_frame = bad_pix_this_frame
            bad_pix_this_frame = np.count_nonzero(S)
            bad_pix_removed += bad_pix_this_frame

            # Report progress.
            if verbose == 2:
                print("Bad pixels removed on iteration %.0f: %.0f" % (iteration_N, bad_pix_this_frame))

            # Correct frames.
            med_filter_image = median_filter(data_frame,size=5)
            data_frame = np.where(S != 0, med_filter_image, data_frame)

            # Increment iteration number and check if condition to stop iterating is hit.
            iteration_N += 1
            if (n != None and iteration_N > n): # if it has hit the iteration cap
                stop_iterating = True
            if (n == None and bad_pix_this_frame == bad_pix_last_frame): # if it has stalled out on finding new outliers
                stop_iterating = True
        
        if verbose == 2:
            print("Finished cleaning frame %.0f in %.0f iterations." % (k, iteration_N-1))
            print("Total pixels corrected: %.0f out of %.0f" % (bad_pix_removed, S.shape[0]*S.shape[1]))
        # Now replace the xarray datasets with the corrected frame and updated dq array.
        obs.images[k] = obs.images[k].where(obs.images[k].values == data_frame,data_frame)
        obs.data_quality[k] = obs.data_quality[k].where(obs.data_quality[k].values == dq,dq)

        if (show_plots == 1 or save_plots == 1) and k == 0:
            plot_exposure([S], min = 1e-3, max = 1, 
                          show_plot=(show_plots>=1), save_plot=(save_plots>=1), 
                          output_dir=output_dir, filename = ['LED_location_of_corrected_pixels_0'])
            
            plot_exposure([obs.images.data[0]],
                          show_plot=(show_plots>=1), save_plot=(save_plots>=1), 
                          output_dir=output_dir, filename = ['LED_after_correction_0'])
        
        elif show_plots == 2 or save_plots == 2:
            plot_exposure([S], min = 1e-3, max = 1, 
                          show_plot=(show_plots==2), save_plot=(save_plots==2), 
                          output_dir=output_dir, filename = ['LED_location_of_corrected_pixels_{}'.format(k)])
            
            plot_exposure([obs.images.data[k]],
                          show_plot=(show_plots==2), save_plot=(save_plots==2), 
                          output_dir=output_dir, filename = ['LED_after_correction_{}'.format(k)])
            
            if k == 0:
                # Additionally plot the noise model and fine structure model, if applicable.
                plot_exposure([noise_model], min = 1e-3, max = 1, 
                              show_plot=(show_plots==2), save_plot=(save_plots==2), 
                              output_dir=output_dir, filename = ['LED_Noise_Model'])
                
                if build_fine_structure:
                    plot_exposure([F], min = 1e-3, max = 1, 
                                  show_plot=(show_plots==2), save_plot=(save_plots==2), 
                                  output_dir=output_dir, filename = ['LED_Fine_Structure_Model'])
    
    if verbose >= 1:
        print("All frames cleaned of spatial outliers by LED.")
    return obs


def build_noise_model(data_frame, readnoise):
    """Builds a noise model for the given data frame, following van Dokkum
    2001 methods.

    Args:
        data_frame (np.array): frame from the images DataSet, used to build
        the noise model.
        readnoise (float): readnoise estimated to be in the data frame.

    Returns:
        np.array: 2D array same size as the data frame, a noise model describing
        noise in the frame.
    """

    noise_model = np.sqrt(median_filter(np.abs(data_frame),size=5)+readnoise**2)
    noise_model[noise_model <= 0] = np.mean(noise_model) # really want to avoid nans
    return noise_model


def subsample_frame(data_frame, factor=2):
    """Subsamples the input frame by the given subsampling factor.

    Args:
        data_frame (np.array): Frame from the images DataSet, used to build
        the noise model.
        factor (int, optional): Factor by which to subsample the array which
        must be >= 2. Defaults to 2.

    Returns:
        np.array: 2D array same shape as data frame, subsampled by factor.
    """
    if factor < 2:
        print("Subsampling factor must be at least 2, forcing factor to 2...")
        factor = 2 # Force factor 2 or more
    factor = int(factor) # Force integer
    
    original_shape = np.shape(data_frame)
    ss_shape = (original_shape[0]*factor,original_shape[1]*factor)
    subsample = np.empty(ss_shape)
    
    # Subsample the array.
    for i in range(ss_shape[0]):
        for j in range(ss_shape[1]):
            try:
                subsample[i,j] = data_frame[int((i+1)/2),int((j+1)/2)]
            except IndexError:
                subsample[i,j] = 0
    return subsample, original_shape


def resample_frame(data_frame, original_shape):
    """Resamples a subsampled array back to the original shape.

    Args:
        data_frame (np.array): subsampled frame from the images DataSet.
        original_shape (tuple of int): original shape of the subsampled array.

    Returns:
        np.array: 2D array with original shape resampled from the data frame.
    """
    resample = np.empty(original_shape)
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            resample[i,j] = 0.25*(data_frame[2*i-1,2*j-1] +
                                  data_frame[2*i-1,2*j]   +
                                  data_frame[2*i,2*j-1]   +
                                  data_frame[2*i,2*j])
    return resample


def build_fine_structure_model(data_frame):
    """Builds a fine structure model for the data frame.

    Args:
        data_frame (np.array): Native resolution data.

    Returns:
        np.array: 2D array of fine structure model.
    """
    F = median_filter(data_frame, size=3) - median_filter(median_filter(data_frame, size=3), size=7)
    F[F <= 0] = np.mean(F) # really want to avoid nans
    return F
