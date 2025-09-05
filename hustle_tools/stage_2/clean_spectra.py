import os
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt

#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def clean_spectra(spec, sigma,
                  verbose = 0):
    """Replace spectral outliers in time with their temporal median.

    Args:
        spec (np.array): spectrum to clean.
        sigma (float): threshold at which to reject a value as an outlier.
        verbose (int, optional): How detailed you want the printed statements
        to be. Defaults to 0.

    Returns:
        np.array: input spectrum cleaned of final outliers.
    """

    if verbose > 0:
        print("Cleaning spectral outliers...")
    
    # Array-ify input spec for use of numpy.
    oneD_spec = np.array(spec)

    # Track outliers removed.
    bad_spex_removed = 0
    n_spex_points = oneD_spec.shape[0]*oneD_spec.shape[1]

    # Iteration stop condition. As long as outliers are being found, we have to keep iterating.
    outlier_found = True
    while outlier_found:
        # Define median spectrum in time and extend its size to include all time.
        med_spec = np.median(oneD_spec,axis=0)
        med_spec = np.array([med_spec,]*oneD_spec.shape[0])
        # Get standard deviation of each point.
        std_spec = np.std(oneD_spec,axis=0)
        std_spec = np.array([std_spec,]*oneD_spec.shape[0])

        # Flag outliers.
        S = np.where(np.abs(oneD_spec-med_spec) > sigma*std_spec, 1, 0)

        # Count outliers found.
        bad_spex_this_step = np.count_nonzero(S)
        bad_spex_removed += bad_spex_this_step

        if bad_spex_this_step == 0:
            # No more outliers found! We can break the loop now.
            outlier_found = False
        
        # Correct outliers and loop once more.
        oneD_spec = np.where(S == 1, med_spec, oneD_spec)

    if verbose > 0:
        print("1D spectral cleaning complete. Removed %.0f spectral outliers from %.0f spectral points." % (bad_spex_removed,
                                                                                                            n_spex_points))

    return oneD_spec


def smooth_spectra(spec, wavelengths, order, orbit_numbers, sigma,
                   verbose = 0, show_plots = 0, save_plots = 0,
                   output_dir = None):
    """Replace spectral outliers in time with their spectral median.

    Args:
        spec (np.array): spectrum to clean.
        wavelengths (np.array): wavelengths for plotting cleaned spectrum.
        order (str, optional): which order this is, for plot title.
        Defaults to "+1".
        orbit_numbers (array-like): orbit number associated with each flux point,
        used in computing the smoothed model.
        sigma (float): threshold at which to reject a value as an outlier.
        verbose (int, optional): How detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): How many plots you want to show. Defaults to 0.
        save_plots (int, optional): How many plots you want to save. Defaults to 0.
        output_dir (str, optional): Where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        np.array: input spectrum cleaned of final outliers.
    """

    if verbose > 0:
        print("Cleaning spectral outliers...")
    
    # Array-ify input spec for use of numpy.
    oneD_spec = np.array(spec)

    # Track outliers removed.
    bad_spex_removed = 0
    n_spex_points = oneD_spec.shape[0]*oneD_spec.shape[1]
    
    # Define kernel size.
    kernel_size = int(0.10*oneD_spec.shape[0])
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3

    # Iteration stop condition. As long as outliers are being found, we have to keep iterating.
    outlier_found = True
    hit_map = np.zeros_like(oneD_spec)
    while outlier_found:
        # Create smoothed model over orbits.
        smoothed_spec = np.empty_like(oneD_spec)
        for i in range(oneD_spec.shape[0]):
            smoothed_spec[i,:] = medfilt(oneD_spec[i,:],kernel_size)
        # Get standard deviation of each point.
        std_spec = np.std(oneD_spec,axis=0)
        std_spec = np.array([std_spec,]*oneD_spec.shape[0])

        # Flag positive outliers.
        S = np.where(oneD_spec-smoothed_spec > sigma*std_spec, 1, 0)
        hit_map += S

        # Count outliers found.
        bad_spex_this_step = np.count_nonzero(S)
        bad_spex_removed += bad_spex_this_step

        if bad_spex_this_step == 0:
            # No more outliers found! We can break the loop now.
            outlier_found = False
        
        # Correct outliers and loop once more.
        oneD_spec = np.where(S == 1, smoothed_spec, oneD_spec)
    
    if (show_plots > 0 or save_plots > 0):
        # Plot hit map at all times.
        plt.figure(figsize = (20, 4))
        plt.imshow(hit_map, origin = 'lower', norm='linear', 
                   vmin = 0, vmax = 1, cmap = 'binary')
        plt.xlabel('Column number (#)')
        plt.ylabel('Exposure number (#)')
        
        if save_plots > 0:
            plot_dir = os.path.join(output_dir, 'plots') 
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir) 
            filedir = os.path.join(plot_dir, f'1Dspec_hitmap_{order}.png')
            plt.savefig(filedir,dpi=300,bbox_inches='tight')

        if show_plots > 0:
            plt.show(block=True)

        plt.close() # save memory

        # Plot hit map summed over time.
        hit_map = np.sum(hit_map,axis=0
                         )
        # bound wavelengths to the region G280 is sensitive to
        ok = (wavelengths>2000) & (wavelengths<8000)

        # initialize plot and plot data that's in the okay range
        plt.figure(figsize = (10, 7))
        plt.plot(wavelengths[ok], hit_map[ok], color='k')
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Flagged (#)')
        plt.title('Pixels flagged as 1D spectral outliers')

        if save_plots > 0:
            plot_dir = os.path.join(output_dir, 'plots') 
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir) 
            filedir = os.path.join(plot_dir, f'1Dspec_totalhits_{order}.png')
            plt.savefig(filedir,dpi=300,bbox_inches='tight')

        if show_plots > 0:
            plt.show(block=True)

        plt.close() # save memory

    if verbose > 0:
        print("1D spectral cleaning complete. Removed %.0f spectral outliers from %.0f spectral points." % (bad_spex_removed,
                                                                                                            n_spex_points))

    return oneD_spec


def running_clean_spectra(spec, wavelengths, order, sigma, kernel_size,
                          verbose = 0, show_plots = 0, save_plots = 0,
                          output_dir = None):
    """Replace spectral outliers in time with their running median.

    Args:
        spec (np.array): spectrum to clean.
        sigma (float): threshold at which to reject a value as an outlier.
        kernel_size (float): kernel for computing the running median as well
        as the running standard deviation.
        verbose (int, optional): How detailed you want the printed statements
        to be. Defaults to 0.

    Returns:
        np.array: input spectrum cleaned of final outliers.
    """

    if verbose > 0:
        print("Cleaning spectral outliers...")
    
    # Array-ify input spec for use of numpy.
    oneD_spec = np.array(spec)

    # Track outliers removed.
    bad_spex_removed = 0
    n_spex_points = oneD_spec.shape[0]*oneD_spec.shape[1]

    # Iteration stop condition. As long as outliers are being found, we have to keep iterating.
    outlier_found = True
    hit_map = np.zeros_like(oneD_spec)
    while outlier_found:
        # Median-filter the spectrum in time over wavelength bins.
        med_spec = np.empty_like(oneD_spec)
        for i in range(oneD_spec.shape[1]):
            med_spec[:,i] = medfilt(oneD_spec[:,i],kernel_size)
        
        # Get running standard deviation of each point.
        std_spec = np.empty_like(oneD_spec)
        for j in range(oneD_spec.shape[0]):
            lbound = min(0,np.abs(int(j-0.5*kernel_size)))
            rbound = max(int(j+0.5*kernel_size),oneD_spec.shape[0])
            for i in range(oneD_spec.shape[1]):
                std_spec[j,i] = np.std(oneD_spec[lbound:rbound,i])
        #std_spec = np.std(oneD_spec,axis=0)
        #std_spec = np.array([std_spec,]*oneD_spec.shape[0])

        # Flag outliers.
        S = np.where(np.abs(oneD_spec-med_spec) > sigma*std_spec, 1, 0)
        hit_map += S

        # Count outliers found.
        bad_spex_this_step = np.count_nonzero(S)
        bad_spex_removed += bad_spex_this_step

        if bad_spex_this_step == 0:
            # No more outliers found! We can break the loop now.
            outlier_found = False
        
        # Correct outliers and loop once more.
        oneD_spec = np.where(S == 1, med_spec, oneD_spec)
    
    if (show_plots > 0 or save_plots > 0):
        # Plot hit map at all times.
        plt.figure(figsize = (20, 4))
        plt.imshow(hit_map, origin = 'lower', norm='linear', 
                   vmin = 0, vmax = 1, cmap = 'binary')
        plt.xlabel('Column number (#)')
        plt.ylabel('Exposure number (#)')
        
        if save_plots > 0:
            plot_dir = os.path.join(output_dir, 'plots') 
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir) 
            filedir = os.path.join(plot_dir, f'1Dspec_hitmap_{order}.png')
            plt.savefig(filedir,dpi=300,bbox_inches='tight')

        if show_plots > 0:
            plt.show(block=True)

        plt.close() # save memory

        # Plot hit map summed over time.
        hit_map = np.sum(hit_map,axis=0
                         )
        # bound wavelengths to the region G280 is sensitive to
        ok = (wavelengths>2000) & (wavelengths<8000)

        # initialize plot and plot data that's in the okay range
        plt.figure(figsize = (10, 7))
        plt.plot(wavelengths[ok], hit_map[ok], color='k')
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Flagged (#)')
        plt.title('Pixels flagged as 1D spectral outliers')

        if save_plots > 0:
            plot_dir = os.path.join(output_dir, 'plots') 
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir) 
            filedir = os.path.join(plot_dir, f'1Dspec_totalhits_{order}.png')
            plt.savefig(filedir,dpi=300,bbox_inches='tight')

        if show_plots > 0:
            plt.show(block=True)

        plt.close() # save memory


    if verbose > 0:
        print("1D spectral cleaning complete. Removed %.0f spectral outliers from %.0f spectral points." % (bad_spex_removed,
                                                                                                            n_spex_points))

    return oneD_spec