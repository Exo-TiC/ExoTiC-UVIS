import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import norm
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import signal
from tqdm import tqdm




def cross_corr(spec, temp_spec, plot = False, trim = 1, fit_window = 5, subpix_width = 0.01):
    """Function to perform cross-correlation of two arrays. Based on ExoTic-JEDI align_spectra.py code

    Args:
        spec (_type_): _description_
        temp_spec (_type_): _description_
        plot (bool, optional): _description_. Defaults to False.
        trim (int, optional): _description_. Defaults to 1.
        fit_window (int, optional): _description_. Defaults to 5.
        subpix_width (float, optional): _description_. Defaults to 0.01.

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
    
    if plot:
        plt.figure()
        plt.plot(corr_lags, corr)

        plt.figure()
        plt.plot(cent_lags, cent_corr)
        plt.plot(cent_lags, p_val)
        plt.show()

    return parab_vtx + trim




def align_spectra(obs, specs, specs_err, trace_x, align = False, ind1 = 0, ind2 = -1, plot_shifts = True):
    """Aligns 1D spectra and uncertainties using cross-correlation

    Args:
        obs (xarray): _description_
        specs (_type_): _description_
        specs_err (_type_): _description_
        trace_x (_type_): _description_
        align (bool, optional): _description_. Defaults to False.
        ind1 (int, optional): _description_. Defaults to 0.
        ind2 (int, optional): _description_. Defaults to -1.
        plot_shifts (bool, optional): _description_. Defaults to True.

    Returns:
        xarray: _description_
    """

    # initialize variables and define median spectrum as template
    align_specs = []
    align_specs_err = []
    x_shifts, y_shifts = [], []
    temp_spec = np.median(specs[:], axis = 0)

    
    # iterate over all spectra
    for i, spec in enumerate(specs):

        # calculate shift wrt template via cross-correlation
        shift = cross_corr(spec[ind1:ind2], temp_spec[ind1:ind2], plot = False, fit_window = 5, subpix_width=0.01)
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


    if plot_shifts:

        plt.figure(figsize = (10, 7))
        plt.plot(obs.exp_time.data, x_shifts, '-o')
        plt.xlabel('Exposure time')
        plt.ylabel('X shift')
        plt.title('Spectrum shift')
        plt.show()
        
        plt.close() # save memory

        colors = plt.cm.rainbow(np.linspace(0, 1, 25))

        plt.figure(figsize = (10, 7))
        for i, spec in enumerate(specs[0:25, ind1:ind2]):
            plt.plot(spec, color = colors[i])

        plt.figure(figsize = (10, 7))
        for i, spec in enumerate(align_specs[0:25, ind1:ind2]):
            plt.plot(spec, color = colors[i])
        plt.show()

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

