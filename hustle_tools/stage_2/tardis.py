from tqdm import tqdm

import numpy as np

from hustle_tools.plotting import plot_exposure, plot_flags_per_time


def time_and_relative_detrending_in_space(obs, trace_x, trace_y, order,
                                          sigmas=[5.0,3.5], flux_threshold=1000, 
                                          windows=[10,5], replacement='tseries',
                                          verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Estimates systematic trends to catch cosmic rays otherwise not detected by any of the Stage 1 cleaning routines.

    Args:
        obs (xarray): obs.images DataSet contains the images.
        trace_x (array-like): x-coordinates of the trace to be corrected.
        trace_y (array-like): y-coordinates of the trace to be corrected.
        order (str): For plot file naming purposes.
        sigmas (lst of float, optional): sigmas to reject outliers at for each window supplied. Defaults to [5.0,3.5].
        flux_threshold (float, optional): median e-/s value at which a pixel's time series will be subjected to this method. Method fails if there is insufficient flux by which to estimate systematics. Defaults to 1000.
        windows (int, optional): median time-series of pixels is computed using pixels +/- window length away from target pixel. Defaults to [10,5].
        replacement (str, optional): if 'tseries', replaces outliers with scaled median time-series of nearby pixels. If 'median', replace outlier pixels with pixel's median value in time. Defaults to 'tseries'.
        verbose (int, optional): how detailed you want the printed statements to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show. Defaults to 0.
        save_plots (int, optional): how many plots you want to save. Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots is greater than 0. Defaults to None.

    Returns:
        xarray: obs with .images cleaned of CRs and with .data_quality updated to indicate where CRs were found.
    """
    # Copy images and define hit map.
    images = obs.images.data.copy()
    hit_map = np.zeros([images.shape[0],images.shape[1],
                        images.shape[2],len(sigmas)])
    
    # Iterate over pixels only if they are suitably bright for this process.
    xfix, yfix = np.where(np.median(images,axis=0)>flux_threshold)

    # Also, only iterate if the coordinates are within the trace region.
    xl,xu,yl,yu = (np.min(trace_y)-35,np.max(trace_y)+35,
                   np.min(trace_x)-10,np.max(trace_x)+10)
    ok = (xfix>xl) & (xfix<xu) & (yfix>yl) & (yfix<yu)
    xfix, yfix = xfix[ok], yfix[ok]
    xl,xu,yl,yu = [round(i) for i in (xl,xu,yl,yu)]

    # Iterate through windows and sigmas.
    for idx,(sigma, window) in enumerate(zip(sigmas,windows)):
        # (re-)Build median t-series using images.
        med_images = images.copy()
        med_images /= np.median(med_images,axis=0)
        for x, y in tqdm(zip(xfix,yfix),total=len(xfix),
                        desc='Identifying CRs in trace region at sigma {:.3f}... Progress:'.format(sigma),
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

            # Update hit map to record where positive outliers were found.
            hit_map[:,x,y,idx] = np.where(detrended_series-med>sigma*sig,1,hit_map[:,x,y,idx])

        # Report results.
        if verbose >= 1:
            print("Identification complete. Pixels identified as %.2f-sigma CRs: %.0f out of %.0f" % (sigma,np.count_nonzero(hit_map[:,:,:,idx]),
                                                                                                      (xu-xl)*(yu-yl)*hit_map.shape[0]))
        
        # Correct for this sigma.
        thits, xhits, yhits = np.where(hit_map[:,:,:,idx]!=0)
        for t,x,y in tqdm(zip(thits,xhits,yhits),total=len(thits),
                        desc='Correcting CRs at {:.3f}-sigma using {}... Progress:'.format(sigma,replacement),
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
                raise ValueError("Replacement method {} not recognized, please update .hustle file and rerun.".format(replacement))
    
    # Report    
    if verbose == 2:
        print("Corrections complete for all windows and sigmas.")
    
    # if true, plot one exposure and draw location of all detected cosmic rays in all exposures
    if save_plots > 0 or show_plots > 0:
        thits, xhits, yhits, ihits = np.where(hit_map == 1)
        plot_exposure([obs.images.data[0,xl:xu,yl:yu], images[0,xl:xu,yl:yu]],
                      title = 'TARDIS removal Example', 
                      show_plot=(show_plots >= 1), save_plot=(save_plots >= 1),
                      output_dir=output_dir, filename = ['TARDIS_order{}_before_correction'.format(order), 'TARDIS_order{}_after_correction'.format(order)])

        plot_exposure([obs.images.data[0,xl:xu,yl:yu]], scatter_data=[yhits-yl, xhits-xl],
                      title = 'Location of corrected pixels', mark_size = 1,
                      show_plot=(show_plots >= 1), save_plot=(save_plots >= 1),
                      output_dir=output_dir, filename = ['TARDIS_order{}_location'.format(order)])
        
        counts_per_frame = [np.count_nonzero(hit_map[i,xl:xu,yl:yu,:]) for i in range(hit_map.shape[0])]
        plot_flags_per_time([obs.exp_time.values,], [counts_per_frame,], style='scatter',
                            title='TARDIS outliers counted per frame',
                            xlabel=['time [mjd]',],
                            ylabel=['counts [#]',],
                            xmin = np.min(obs.exp_time.values), xmax = np.max(obs.exp_time.values),
                            ymin = 0.995*np.min(counts_per_frame), ymax = 1.005*np.max(counts_per_frame),
                            show_plot=(show_plots>=1),save_plot=(save_plots>=1),
                            filename=['TARDIS_order{}_outliers_per_frame'.format(order),],output_dir=output_dir)

    # if true, check each exposure separately
    if save_plots == 2 or show_plots == 2:
        for i in range(len(images)):
            xhits, yhits, ihits = np.where(hit_map[i,:,:,:] == 1)
            plot_exposure([obs.images.data[i,xl:xu,yl:yu]], scatter_data=[yhits-yl, xhits-xl],
                          title = 'Location of corrected pixels in frame {}'.format(i), mark_size = 1,
                          show_plot=(show_plots == 2), save_plot=(save_plots == 2),
                          output_dir=output_dir, filename = [f'TARDIS_order{order}_location_frame{i}'])
    # collapse hit map onto one axis
    new_hit_map = np.empty_like(images)
    for i in range(hit_map.shape[3]):
        new_hit_map += hit_map[:,:,:,i]
    new_hit_map[new_hit_map > 0] = 1

    # modify original images and dq
    obs.images.data = images
    obs.data_quality.data = np.where(new_hit_map != 0, new_hit_map, obs.data_quality.data)

    return obs
