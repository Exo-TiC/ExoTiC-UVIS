from tqdm import tqdm

import numpy as np

from hustle_tools.plotting import plot_exposure, plot_flags_per_time


def fixed_iteration_rejection(obs, sigmas=[10,10], replacement=None,
                              verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Iterates a fixed number of times using a different sigma at each
    iteration to reject cosmic rays.

    Args:
        obs (xarray): obs.images DataSet contains the images.
        sigmas (list, optional): sigma to use for each iteration. len(sigmas)
        is the number of iterations that will be run. Defaults to [10,10].
        replacement (int, optional): if None, replace outlier pixels with
        median in time. If int, replace with median of int values either side
        in time. Defaults to None.
        verbose (int, optional): how detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show. Defaults to 0.
        save_plots (int, optional): how many plots you want to save. Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        xarray: obs with .images cleaned of CRs and with .data_quality updated
        to indicate where CRs were found.
    """
    # Copy images and define hit map.
    images = obs.images.data.copy()
    hit_map = np.zeros_like(images)

    # Iterate over each sigma.
    for j, sigma in tqdm(enumerate(sigmas), total=len(sigmas),
                         desc='Iterating with fixed sigmas to remove CRs... Progess:',
                         disable=(verbose < 1)):
        # Get the median time frame and std as a reference.
        med = np.median(images,axis=0)
        std = np.std(images,axis=0)

        # Track outliers flagged by this sigma.
        bad_pix_this_sigma = 0

        # Then check over frames and see where outliers are.
        for k in tqdm(range(images.shape[0]),
                      desc = "Correcting for %.0fth sigma... Progress:" % j,
                      disable=(verbose < 2)):
            # Get the frame and dq array as np.array objects so we can operate on them.
            d = images[k]
            S = np.where(np.abs(d - med) > sigma*std, 1, 0)
            
            # Report where data quality flags should be added and count pixels to be replaced.
            dq = np.where(S != 0, 1, 0)
            hit_map[k,:,:] += dq
            bad_pix_this_frame = np.count_nonzero(S)
            bad_pix_this_sigma += bad_pix_this_frame

            # If replacement is not None, custom replacement.
            correction = med
            if replacement:
                # Take the median of the frames that are +/- replacement away from the current frame.
                l = k - replacement
                r = k + replacement
                # Cut at edges.
                if l < 0:
                    l = 0
                if r > images.shape[0]:
                    r = images.shape[0]
                correction = np.median(images[l:r,:,:],axis=0)
            # Correct frame and replace in images.
            images[k] = np.where(S == 1, correction, d)
        
        if verbose == 2:
            print("Bad pixels removed on iteration %.0f with sigma %.2f: %.0f" % (j, sigma, bad_pix_this_sigma))
    
    # Correct hit map.
    hit_map[hit_map != 0] = 1
    if verbose >= 1:
        print("Fixed iterations complete. Total pixels corrected: %.0f out of %.0f" % (np.count_nonzero(hit_map),
                                                                                       hit_map.shape[0]*hit_map.shape[1]*hit_map.shape[2]))

    # if true, plot one exposure and draw location of all detected cosmic rays in all exposures
    if save_plots > 0 or show_plots > 0:
        thits, xhits, yhits = np.where(hit_map == 1)
        plot_exposure([obs.images.data[0], images[0]],
                      title = 'Temporal Bad Pixel removal Example', 
                      show_plot=(show_plots >= 1), save_plot=(save_plots >= 1),
                      output_dir=output_dir, filename = ['CR-fixed_before_correction', 'CR-fixed_after_correction'])

        plot_exposure([obs.images.data[0]], scatter_data=[yhits, xhits],
                      title = 'Location of corrected pixels', mark_size = 1,
                      show_plot=(show_plots >= 1), save_plot=(save_plots >= 1),
                      output_dir=output_dir, filename = ['CR-fixed_location'])
        
        counts_per_frame = [np.count_nonzero(hit_map[i,:,:]) for i in range(hit_map.shape[0])]
        plot_flags_per_time([obs.exp_time.values,], [counts_per_frame,], style='scatter',
                            title='Temporal outliers counted per frame',
                            xlabel=['time [mjd]',],
                            ylabel=['counts [#]',],
                            xmin = np.min(obs.exp_time.values), xmax = np.max(obs.exp_time.values),
                            ymin = 0.995*np.min(counts_per_frame), ymax = 1.005*np.max(counts_per_frame),
                            show_plot=(show_plots>=1),save_plot=(save_plots>=1),
                            filename=['CR-fixed_outliers_per_frame',],output_dir=output_dir)

    # if true, check each exposure separately
    if save_plots == 2 or show_plots == 2:
        for i in range(len(images)):
            xhits, yhits = np.where(hit_map[i] == 1)
            plot_exposure([obs.images.data[i]], scatter_data=[yhits, xhits],
                          title = 'Location of corrected pixels in frame {}'.format(i), mark_size = 1,
                          show_plot=(show_plots == 2), save_plot=(save_plots == 2),
                          output_dir=output_dir, filename = [f'CR-fixed_location_frame{i}'])
            
    # modify original images and dq
    obs.images.data = images
    obs.data_quality.data = np.where(hit_map != 0, hit_map, obs.data_quality.data)

    return obs


def array1D_clip(array, threshold = 3.5):
    """Function to detect and replace outliers in a 1D array above or below
    a certain sigma threshold imposed.

    Args:
        array (np.array): pixel time series to be cleaned for outliers.
        threshold (float, optional): threshold at which to call a value
        an outlier. Defaults to 3.5.

    Returns:
        np.array: cleaned time series and mask marking where outliers were found.
    """
    
    # define outlier flag and mask
    found_outlier = 1
    mask = np.ones_like(array).astype(bool)

    # iterate while flag is true
    while found_outlier:
        
        # compute median and std of masked array
        n_hits = np.sum(mask)
        median = np.median(array[mask])
        sigma = np.std(array[mask])

        # mask values below threshold
        mask = np.abs(array - median) < threshold*sigma     
        found_outlier = n_hits - np.sum(mask)
    
    # replace masked values with median
    array[~mask] = median

    return array, ~mask


def free_iteration_rejection(obs, threshold = 3.5,
                             verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Function to replace outliers in the temporal dimension.

    Args:
        obs (xarray): obs.images DataSet contains the images.
        threshold (float, optional): sigma at which to reject outliers until
        no more are found at this level. Defaults to 3.5.
        verbose (int, optional): how detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show. Defaults to 0.
        save_plots (int, optional): how many plots you want to save. Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        xarray: obs with .images cleaned of CRs and with .data_quality updated
        to indicate where CRs were found.
    """
    
    # copy images and define hit map
    images = obs.images.data.copy()
    hit_map = np.zeros_like(images)
    
    # iterate over all rows
    for i in tqdm(range(obs.dims['x']), desc = 'Removing cosmic rays and bad pixels... Progress:'):
        #iterate over all columns
        for j in range(obs.dims['y']):
            # check that sum of pixel along temporal dimension is non-zero (i.e., that the pixel is inside the subarray)
            if np.sum(images[:, i, j]):
                _, hit_map[:, i, j] = array1D_clip(images[:, i, j], threshold)
    
    # if true, plot one exposure and draw location of all detected cosmic rays in all exposures
    if save_plots > 0 or show_plots > 0:
        thits, xhits, yhits = np.where(hit_map == 1)
        plot_exposure([obs.images.data[0], images[0]], min = 1e0, 
                      title = 'Temporal Bad Pixel removal Example', 
                      show_plot=(show_plots > 0), save_plot=(save_plots > 0),
                      output_dir=output_dir,
                      filename = ['CR-free_before_correction', 'CR-free_after_correction'])

        plot_exposure([obs.images.data[0]], scatter_data=[yhits, xhits], min = 1e0, 
                      title = 'Location of corrected pixels', mark_size = 1,
                      show_plot=(show_plots > 0), save_plot=(save_plots > 0),
                      output_dir=output_dir, filename = ['CR-free_location'])
        
        counts_per_frame = [np.count_nonzero(hit_map[i,:,:]) for i in range(hit_map.shape[0])]
        plot_flags_per_time([obs.exp_time.values,], [counts_per_frame,], style='scatter',
                            title='Temporal outliers counted per frame',
                            xlabel=['time [mjd]',],
                            ylabel=['counts [#]',],
                            xmin = np.min(obs.exp_time.values), xmax = np.max(obs.exp_time.values),
                            ymin = 0.995*np.min(counts_per_frame), ymax = 1.005*np.max(counts_per_frame),
                            show_plot=(show_plots > 0),save_plot=(save_plots > 0),
                            filename=['CR-free_outliers_per_frame',],output_dir=output_dir)

    # if true, check each exposure separately
    if save_plots == 2 or show_plots == 2:
        for i in range(len(images)):
            xhits, yhits = np.where(hit_map[i] == 1)
            plot_exposure([obs.images.data[i]], scatter_data=[yhits, xhits],
                          title = 'Location of corrected pixels in frame {}'.format(i), mark_size = 1,
                          show_plot=(show_plots == 2), save_plot=(save_plots == 2),
                          output_dir=output_dir, filename = [f'CR-free_location_frame{i}'])
    
    # modify original images and dq
    obs.images.data = images
    obs.data_quality.data = np.where(hit_map != 0, hit_map, obs.data_quality.data)

    # Report bad pixels.
    if verbose >= 1:
        print("Free iterations complete. Total pixels corrected: %.0f out of %.0f" % (np.count_nonzero(hit_map),
                                                                                      hit_map.shape[0]*hit_map.shape[1]*hit_map.shape[2]))

    return obs


def detrend_outlier_rejection(obs, sigma=3.5, flux_threshold=1000, 
                              window=10, replacement='tseries',
                              verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Estimates systematic trends to catch cosmic rays otherwise not detected
    by either of the other available time outlier rejection methods.

    Args:
        obs (xarray): obs.images DataSet contains the images.
        sigma (float, optional): sigma to reject outliers at. Defaults to 3.5.
        flux_threshold (float, optional): median e-/s value at which a pixel's
        time series will be subjected to this method. Method fails if there is
        insufficient flux by which to estimate systematics. Defaults to 1000.
        window (int, optional): median time-series of pixels is computed using
        pixels +/- window length away from target pixel. Defaults to 10.
        replacement (str, optional): if 'tseries', replaces outliers with scaled
        median time-series of nearby pixels. If 'median', replace outlier pixels
        with pixel's median value in time. Defaults to 'tseries'.
        verbose (int, optional): how detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show. Defaults to 0.
        save_plots (int, optional): how many plots you want to save. Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        xarray: obs with .images cleaned of CRs and with .data_quality updated
        to indicate where CRs were found.
    """
    # Copy images, get median-normalized t-series, and define hit map.
    images = obs.images.data.copy()
    med_images = obs.images.data.copy()
    med_images /= np.median(med_images,axis=0)
    hit_map = np.zeros_like(images)

    # Iterate over each suitably-bright pixel.
    xfix, yfix = np.where(np.median(images,axis=0)>flux_threshold)
    for x, y in tqdm(zip(xfix,yfix),total=len(xfix),
                     desc='Identifying CRs in areas with strong systematics... Progress:',
                     disable=(verbose < 1)):
        # Do not attempt corrections if y is too close to the detector edge.
        proximity_alert = (y-window<0) or (y+window>med_images.shape[2])
        if proximity_alert:
            continue

        # Get local median time-series.
        med_series = np.median(med_images[:,x,y-window:y+window],axis=1)

        # Detrend target time-series and measure statistics.
        detrended_series = med_images[:,x,y]/med_series
        med = np.median(detrended_series)
        sig = np.std(detrended_series)

        # Update hit map to record where outliers were found.
        hit_map[:,x,y] = np.where(np.abs(detrended_series-med)>sigma*sig,1,hit_map[:,x,y])
    
    # Report results.
    if verbose >= 1:
        print("Identification complete. Pixels identified as %.2f-sigma CRs: %.0f out of %.0f" % (sigma,np.count_nonzero(hit_map),
                                                                                                  hit_map.shape[0]*hit_map.shape[1]*hit_map.shape[2]))

    # Iterate through hits in time and space and replace as requested.
    thits, xhits, yhits = np.where(hit_map!=0)
    for t,x,y in tqdm(zip(thits,xhits,yhits),total=len(thits),
                      desc='Correcting CRs using {}... Progress:'.format(replacement),
                      disable=(verbose < 1)):
        if replacement == 'tseries':
            # Recompute median time series.
            med_series = np.median(med_images[:,x,y-window:y+window],axis=1)

            # Get scale factor from actual data.
            scale_factor = np.median(images[:,x,y],axis=0)

            # Update data.
            images[t,x,y] = scale_factor*med_series[t]

        elif replacement == 'median':
            # Update data.
            images[t,x,y] = np.median(images[:,x,y])
        
        else:
            raise ValueError("Replacement method {} not recongized, please update .hustle file and rerun.".format(replacement))
    
    # Report    
    if verbose == 2:
        print("Corrections complete.")
    
    # if true, plot one exposure and draw location of all detected cosmic rays in all exposures
    if save_plots > 0 or show_plots > 0:
        thits, xhits, yhits = np.where(hit_map == 1)
        plot_exposure([obs.images.data[0], images[0]],
                      title = 'Temporal Bad Pixel removal Example', 
                      show_plot=(show_plots >= 1), save_plot=(save_plots >= 1),
                      output_dir=output_dir, filename = ['CR-detrend_before_correction', 'CR-detrend_after_correction'])

        plot_exposure([obs.images.data[0]], scatter_data=[yhits, xhits],
                      title = 'Location of corrected pixels', mark_size = 1,
                      show_plot=(show_plots >= 1), save_plot=(save_plots >= 1),
                      output_dir=output_dir, filename = ['CR-detrend_location'])
        
        counts_per_frame = [np.count_nonzero(hit_map[i,:,:]) for i in range(hit_map.shape[0])]
        plot_flags_per_time([obs.exp_time.values,], [counts_per_frame,], style='scatter',
                            title='Temporal outliers counted per frame',
                            xlabel=['time [mjd]',],
                            ylabel=['counts [#]',],
                            xmin = np.min(obs.exp_time.values), xmax = np.max(obs.exp_time.values),
                            ymin = 0.995*np.min(counts_per_frame), ymax = 1.005*np.max(counts_per_frame),
                            show_plot=(show_plots>=1),save_plot=(save_plots>=1),
                            filename=['CR-detrend_outliers_per_frame',],output_dir=output_dir)

    # if true, check each exposure separately
    if save_plots == 2 or show_plots == 2:
        for i in range(len(images)):
            xhits, yhits = np.where(hit_map[i] == 1)
            plot_exposure([obs.images.data[i]], scatter_data=[yhits, xhits],
                          title = 'Location of corrected pixels in frame {}'.format(i), mark_size = 1,
                          show_plot=(show_plots == 2), save_plot=(save_plots == 2),
                          output_dir=output_dir, filename = [f'CR-detrend_location_frame{i}'])
            
    # modify original images and dq
    obs.images.data = images
    obs.data_quality.data = np.where(hit_map != 0, hit_map, obs.data_quality.data)

    return obs
