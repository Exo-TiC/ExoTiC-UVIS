import os
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import norm
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import signal




def cross_corr(spec, temp_spec, order, i, trim = 1, fit_window = 5, subpix_width = 0.01,
               verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Function to perform cross-correlation of two arrays.
    Based on ExoTic-JEDI align_spectra.py code

    Args:
        spec (_type_): _description_
        temp_spec (_type_): _description_
        order (str): for labelling plots correctly.
        i (float): for labelling plots correctly.
        trim (int, optional): _description_. Defaults to 1.
        fit_window (int, optional): _description_. Defaults to 5.
        subpix_width (float, optional): _description_. Defaults to 0.01.
        verbose (int, optional): How detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): How many plots you want to show. Defaults to 0.
        save_plots (int, optional): How many plots you want to save. Defaults to 0.
        output_dir (str, optional): Where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        _type_: _description_
    """

    xvals = np.arange(0, spec.shape[0])

    spec_trim = spec[trim : -trim]
    xvals_trim = xvals[trim : -trim]
    
    interp_spec = interp1d(np.arange(0, spec_trim.shape[0]), spec_trim, kind = 'linear')
    highres_spec = interp_spec(np.arange(0, spec_trim.shape[0] - 1, subpix_width))
    interp_temp = interp1d(np.arange(temp_spec.shape[0]), temp_spec, kind = 'linear') 
    highres_temp = interp_temp(np.arange(0, temp_spec.shape[0] - 1, subpix_width))

    highres_spec -= np.linspace(highres_spec[0], highres_spec[-1], highres_spec.shape[0])
    highres_temp -= np.linspace(highres_temp[0], highres_temp[-1], highres_temp.shape[0])
    highres_spec = (highres_spec - np.mean(highres_spec))/np.std(highres_spec)
    highres_temp = (highres_temp - np.mean(highres_temp))/np.std(highres_temp)

    # cross-correlation
    corr = signal.correlate(highres_spec, highres_temp, mode = 'full')
    corr_lags = signal.correlation_lags(highres_spec.size, highres_temp.size, mode = 'full')
    max_pix = np.argmax(corr)

    # fit parabola
    cent_lags = corr_lags[max_pix - fit_window: max_pix + fit_window + 1].copy()
    cent_corr = corr[max_pix - fit_window: max_pix + fit_window + 1].copy()

    # normalize
    cent_corr -= np.min(cent_corr)
    cent_corr /= np.max(cent_corr)

    p_coeffs = np.polyfit(cent_lags, cent_corr, deg = 2)
    p_val = np.polyval(p_coeffs, cent_lags)

    parab_vtx = -p_coeffs[1] / (2 * p_coeffs[0]) * subpix_width
    
    if (show_plots == 2 or save_plots == 2):
        plt.figure()
        plt.plot(corr_lags, corr)

        plt.figure()
        plt.plot(cent_lags, cent_corr)
        plt.plot(cent_lags, p_val)

        if save_plots == 2:
            stagedir = os.path.join(output_dir, "stage2/plots")
            if not os.path.exists(stagedir):
                os.makedirs(stagedir)
            plt.savefig(os.path.join(stagedir,'s2_cross_corr_order{}_f{}.png'.format(order,i)),
                        dpi=300,bbox_inches='tight')
        if show_plots == 2:
            plt.show(block=True)
        plt.close()
        plt.close() # save memory

    return parab_vtx + trim




def align_spectra(obs, specs, specs_err, order, trace_x, align = False, ind1 = 0, ind2 = -1,
                  verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Aligns 1D spectra and uncertainties using cross-correlation.

    Args:
        obs (xarray): used to attach the spectra to an xarray object.
        specs (np.array): array of 1D spectra.
        specs_err (np.array): array of 1D spectral uncertainties.
        order (str): for labelling plots correctly.
        trace_x (np.array): x positions of the trace solution.
        align (bool, optional): whether to apply the alignment to the spectra.
        Defaults to False.
        ind1 (int, optional): _description_. Defaults to 0.
        ind2 (int, optional): _description_. Defaults to -1.
        verbose (int, optional): How detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): How many plots you want to show. Defaults to 0.
        save_plots (int, optional): How many plots you want to save. Defaults to 0.
        output_dir (str, optional): Where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        xarray: aligned obs spectra.
    """
    # initialize variables and define median spectrum as template
    align_specs = []
    align_specs_err = []
    x_shifts, y_shifts = [], []
    # align only on the intended wavelengths of analysis
    ok = (trace_x>2000) & (trace_x<8000)
    temp_spec = np.median(specs[:,ok], axis = 0)

    
    # iterate over all spectra
    for i, spec in tqdm(enumerate(specs),
                        desc='Aligning spectra in order {}... Progress:'.format(order),
                        disable=(verbose==0)):

        # calculate shift wrt template via cross-correlation
        shift = cross_corr(spec[ok][ind1:ind2], temp_spec[ind1:ind2], order, i,
                           trim = 1, fit_window = 5, subpix_width=0.01,
                           verbose = 0, show_plots = 0,
                           save_plots = 0, output_dir = None)
        x_shifts.append(shift)
        
        # if true, correct the spectrum with the computed shift
        if align:
            shift_tracex = trace_x + shift
            interp_spec = interp1d(trace_x, spec, kind = 'linear', fill_value = 'extrapolate')
            align_specs.append(interp_spec(shift_tracex))

            interp_err = interp1d(trace_x, specs_err[i], kind = 'linear', fill_value = 'extrapolate')
            align_specs_err.append(interp_err(shift_tracex))

        else: 
            align_specs.append(spec)
            align_specs_err.append(specs_err[i])
    
    align_specs = np.array(align_specs)
    align_specs_err = np.array(align_specs_err)


    if (save_plots > 0 or show_plots > 0):
        plt.figure(figsize = (10, 7))
        plt.plot(obs.exp_time.data, x_shifts, '-o')
        plt.xlabel('Exposure time')
        plt.ylabel('X shift')
        plt.title('Spectrum shift')
        if save_plots > 0:
            stagedir = os.path.join(output_dir, "stage2/plots")
            if not os.path.exists(stagedir):
                os.makedirs(stagedir)
            plt.savefig(os.path.join(stagedir,'s2_cross_corr_order{}.png'.format(order)),
                        dpi=300,bbox_inches='tight')
        if show_plots > 0:
            plt.show(block=True)
        
        plt.close() # save memory

        colors = plt.cm.rainbow(np.linspace(0, 1, 25))

        plt.figure(figsize = (10, 7))
        for i, spec in enumerate(specs[0:25, ind1:ind2]):
            plt.plot(spec, color = colors[i])

        plt.figure(figsize = (10, 7))
        for i, spec in enumerate(align_specs[0:25, ind1:ind2]):
            plt.plot(spec, color = colors[i])

        if save_plots > 0:
            stagedir = os.path.join(output_dir, "stage2/plots")
            if not os.path.exists(stagedir):
                os.makedirs(stagedir)
            plt.savefig(os.path.join(stagedir,'s2_shifted_spec_order{}.png'.format(order)),
                        dpi=300,bbox_inches='tight')
        if show_plots > 0:
            plt.show(block=True)

        plt.close() # save memory
        plt.close() # save memory
    
    return align_specs, align_specs_err, np.array(x_shifts)




def align_profiles(obs, trace_x, traces_y, width = 25, plot_median = False):


    """
    
    Function to find the y displacement of the psf

    """

    # copy images and errors
    images = obs.images.data.copy()
    exp_times = obs.exp_time.data
    y_shifts = []
    
    # iterate over all pixels in trace
    for j, pix in enumerate(tqdm(trace_x, desc = 'Computing profile displacements... Progress:')):
  
        y_shift = []
    
        # get trace position
        trace_y = traces_y[j]

        # define lower and upper limits where the sum is performed
        low_val = int(trace_y - width)
        up_val = int(trace_y + width)

        # get profiles
        yvals = np.arange(low_val, up_val)
        profs = images[:, low_val:up_val, int(pix)]

        # get template profile:
        temp_prof = np.median(profs, axis = 0)

        for i, prof in enumerate(profs):
            shift = cross_corr(prof, temp_prof, plot = False, trim = 1)
            y_shift.append(shift)
        
        y_shifts.append(y_shift)
  
    y_shifts = np.array(y_shifts).transpose()

    if plot_median:
        plt.figure(figsize = (10, 7))
        plt.scatter(exp_times, np.median(y_shifts, axis = 1))
        plt.xlabel('Exposure time')
        plt.ylabel('Y displacement')
        plt.show()
        
    return y_shifts

