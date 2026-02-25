from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt2d
from scipy.ndimage import maximum_filter

from hustle_tools.plotting import plot_exposure, plot_flags_per_time


def correct_hst_flags(obs, flags=[4,16,4096], replace=True,
                      verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Uses HST data quality flag information to correct chosen flags.

    Args:
        obs (xarray): obs.images contains the dataset we are cleaning.
        flags (list, optional): FLAG values to correct. See HST WFC3/UVIS G280 handbook for details. Defaults to [4,16,4096].
        replace (bool, optional): whether to replace or simply mask the flagged pixels. Defaults to True.
        verbose (int, optional): how detailed you want the printed statements to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show. Defaults to 0.
        save_plots (int, optional): how many plots you want to save. Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots is greater than 0. Defaults to None.

    Returns:
        xarray: obs with images cleaned and data quality flags updated.
    """
    # Copy images and define hit map.
    images = obs.images.data.copy()
    images = np.array(images)
    hit_map = np.zeros_like(images)

    # Have all HST flags on hand for reference
    hst_flags = [2**i for i in range(1,15)]

    # Iterate over each flag.
    for target_flag in sorted(flags,reverse=True):
        # Target largest flags first and move to successively lower.
        hst_dq = obs.hst_dq.data.copy()
        for i in tqdm(range(images.shape[0]), desc = 'Correcting HST flag {}... Progress:'.format(target_flag),
                      disable=(verbose==0)):
            # Fetch relevant data.
            dq = hst_dq[i]

            # Subtract off flags larger than the current flag if present.
            for hst_flag in [x for x in sorted(hst_flags,reverse=True) if x > target_flag]:
                dq[dq>=hst_flag] -= hst_flag
            
            # Anything greater than or equal to the desired flag at this point has our flag in it.
            dq[dq<target_flag] = 0

            # Anything not zero has the target flag in it and must be corrected.
            if replace:
                if target_flag >= 4096:
                    # Needs to be corrected with time median.
                    images[i] = np.where(dq>0,np.median(images,axis=0),images[i])
                elif target_flag == 256:
                    # Saturated pixel. Needs to be zero in all time,
                    # and needs to be bloomed.
                    dq = maximum_filter(dq,size=7)
                    for k in range(images.shape[0]):
                        images[k] = np.where(dq>0,0,images[k])
                else:
                    # Needs to be corrected with spatial median.
                    images[i] = np.where(dq>0,medfilt2d(images[i],kernel_size=7),images[i])
            else:
                # Simply report this to the bad_pix map.
                obs.badpix_mask.values[i,:,:] = np.where(dq>0,True,obs.badpix_mask.values[i,:,:])
            
            # Update hit map.
            hit_map[i,:,:] = np.where(dq>0,1,hit_map[i,:,:])
    
    # Report results.
    if verbose >= 1:
        print("HST flag corrections complete. Total pixels corrected: %.0f out of %.0f" % (np.count_nonzero(hit_map),
                                                                                           hit_map.shape[0]*hit_map.shape[1]*hit_map.shape[2]))
    
    # if true, plot one exposure and draw location of all corrected flagged pixels in all exposures
    if save_plots > 0 or show_plots > 0:
        thits, xhits, yhits = np.where(hit_map == 1)
        plot_exposure([obs.images.data[0], images[0]],
                      title = 'HST Flag Correction', 
                      show_plot=(show_plots >= 1), save_plot=(save_plots >= 1),
                      output_dir=output_dir, filename = ['HST-flags_before_correction', 'HST-flags_after_correction'])

        plot_exposure([obs.images.data[0]], scatter_data=[yhits, xhits],
                      title = 'Location of corrected pixels', mark_size = 1,
                      show_plot=(show_plots >= 1), save_plot=(save_plots >= 1),
                      output_dir=output_dir, filename = ['HST-flags_location'])
        
        counts_per_frame = [np.count_nonzero(hit_map[i,:,:]) for i in range(hit_map.shape[0])]
        plot_flags_per_time([obs.exp_time.values,], [counts_per_frame,], style='scatter',
                            title='HST flags corrected per frame',
                            xlabel=['time [mjd]',],
                            ylabel=['flags [#]',],
                            xmin = np.min(obs.exp_time.values), xmax = np.max(obs.exp_time.values),
                            ymin = 0.995*np.min(counts_per_frame), ymax = 1.005*np.max(counts_per_frame),
                            show_plot=(show_plots>=1),save_plot=(save_plots>=1),
                            filename=['HST-flags_outliers_per_frame',],output_dir=output_dir)

    # if true, check each exposure separately
    if save_plots == 2 or show_plots == 2:
        for i in range(len(images)):
            xhits, yhits = np.where(hit_map[i] == 1)
            plot_exposure([obs.images.data[i]], scatter_data=[yhits, xhits],
                          title = 'Location of corrected pixels in frame {}'.format(i), mark_size = 1,
                          show_plot=(show_plots == 2), save_plot=(save_plots == 2),
                          output_dir=output_dir, filename = [f'HST-flags_location_frame{i}'])
            
    # modify original images and dq
    obs.images.data = images
    obs.data_quality.data = np.where(hit_map != 0, hit_map, obs.data_quality.data)
        
    return obs
