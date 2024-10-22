import numpy as np

def bin_light_curves(oneDspec, oneDerr, wave, time, order="+1",
                     reject_bad_cols=True, bad_col_threshold=0.01,
                     method='columns', Nbin=50, wavelength_bins=np.arange(2000,8100,100),
                     verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Bins light curves using the given binning scheme.

    Args:
        oneDspec (np.array): 1D spectra for given order as a function of time,
        with indices time and wavelength.
        oneDerr (np.array): 1D uncertainties for given order as a function of time,
        with indices time and wavelength.
        wave (np.array): wavelength solution for spectra.
        time (np.array): timestamps of the exposures in the spectra.
        order (str): for labelling plots correctly.
        reject_bad_cols (bool): if True, mask columns deemed too noisy.
        bad_col_threshold (float): the lower the threshold, the less noisiness
        we tolerate being in our columns.
        method (str, optional): binning method, can be either 'columns' or
        'wavelengths'. Defaults to 'columns'.
        Nbin (int, optional): if method is 'columns', how many columns to bin.
        Defaults to 50.
        wavelength_bins (list or np.array, optional): if method is 'wavelengths',
        the edges of each bin in angstroms. Defaults to np.arange(2000,8100,100).
        verbose (int, optional): How detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): How many plots you want to show. Defaults to 0.
        save_plots (int, optional): How many plots you want to save. Defaults to 0.
        output_dir (str, optional): Where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        np.array,np.array,np.array: binned light curves with central wavelengths
        and wavelength bin sizes.
    """
    # Column binning method.
    if method == 'columns':
        # do stuff
        pass

    # Wavelength binning method.
    if method == 'wavelengths':
        # do stuff
        pass
    return 'yay'