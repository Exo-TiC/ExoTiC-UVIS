import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def standard_extraction(obs, halfwidth, trace_x, trace_y,
                        verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Extracts a standard 1D spectrum from every image using no weighting.

    Args:
        obs (xarray): obs.images contains the data.
        halfwidth (int): "halfwidth" of the extraction aperture, which spans
        from A-hw to A+hw where A is the index of the central row.
        trace_x (np.array): x positions of the pixels in the trace solution.
        trace_y (np.array): y positions of the pixels in the trace solution.
        verbose (int, optional): How detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): How many plots you want to show. Defaults to 0.
        save_plots (int, optional): How many plots you want to save. Defaults to 0.
        output_dir (str, optional): Where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        np.array,np.array: 1D spectrum and errors.
    """
    # Define traces array.
    traces = []
    err_traces = []

    # Iterate over frames.
    for k in tqdm(range(obs.images.shape[0]),desc='Extracting standard spectra... Progress:',
                  disable=(verbose==0)):
        # Get array values and error values of the current frame.
        frame = obs.images[k].values
        err = obs.errors[k].values

        # Pull out just the trace from the frame.
        trace = get_trace(frame, halfwidth, trace_x, trace_y[k])
        err_trace = get_trace(err, halfwidth, trace_x, trace_y[k])

        traces.append(trace)
        err_traces.append(err_trace)
        
    # Numpy array it.
    traces = np.array(traces)
    err_traces = np.array(err_traces)

    # Store the 1D spectra and errors
    oneD_spec = []
    spec_err = []

    # Extract 1D spectrum using the standard method.
    for k in range(traces.shape[0]):
        err = box(err_traces[k,:,:])
        flx = box(traces[k,:,:])
        oneD_spec.append(flx)
        spec_err.append(err)
    
    return np.array(oneD_spec), np.array(spec_err)

def get_trace(frame, halfwidth, xs, ys):
    """Short function to pull a trace region from a frame using the given
    solution and halfwidth.

    Args:
        frame (np.array): one frame from obs.images Dataset.
        halfwidth (int): halfwidth of extraction. Pulls pixels from A-hw to
        A+hw where A is the central row index.
        xs (np.array): x positions of the pixels in the trace solution.
        ys (np.array): y positions of the pixels in the trace solution.

    Returns:
        np.array: dispersion profiles from the trace.
    """
    dispersion_profiles = []
    for i,x in enumerate(xs):
        x_pos = int(x)
        y_pos = range(int(ys[i])-halfwidth,int(ys[i])+halfwidth+1)
        dispersion_profile = frame[y_pos[0]:y_pos[-1],x_pos]
        dispersion_profiles.append(dispersion_profile)
    return np.array(dispersion_profiles)

def box(trace):
    """The simplest extraction method, this routine sums the trace along
    columns without any weighting.

    Args:
        trace (np.array): one frame in time showing the trace at integration k.

    Returns:
        np.array: 1D array of the unweighted spectrum from that trace.
    """
    return np.nansum(trace,axis=1)

def determine_ideal_halfwidth(obs, order, trace_x, trace_y, wavs, indices=([0,10],[-10,-1]),
                              verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Extracts multiple standard white light curves and determines the half-width
    that minimizes scatter out of transit/eclipse.

    Args:
        obs (xarray): obs.images DataSet contains the images.
        order (str): used to label plots appropriately.
        trace_x (np.array): x positions of the pixels in the trace solutions.
        trace_y (np.array): y positions of the pixels in the trace solutions.
        wavs (np.array): wavelength solution for each image.
        indices (tuple, optional): indices that define the out-of-transit/eclipse
        flux, for which scatter is measured. Defaults to ([0,10],[-10,-1]).
        verbose (int, optional): How detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): How many plots you want to show. Defaults to 0.
        save_plots (int, optional): How many plots you want to save. Defaults to 0.
        output_dir (str, optional): Where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        int: half-width integer that minimizes scatter.
    """
    # Initialiaze hws and residuals lists.
    tested_hws, reses = [i for i in range(5,26)], []

    # Test each half-width and measure its scatter.
    wlcs,ts = [],[]
    for hw in tqdm(tested_hws, desc='Testing half-widths to minimize scatter... Progress:',
                   disable=(verbose==0)):
        # Get the 1D spectra.
        oneD_spec, oneD_err = standard_extraction(obs, hw, trace_x, trace_y)
        # Bin into a median-normalized white light curve on valid wavelength range.
        ok = (wavs>2000) & (wavs<8000)
        WLC = np.nansum(oneD_spec[:,ok],axis=1)
        WLC /= np.nanmedian(WLC)
        # Truncate to just the range of out-of-transit/eclipse for each set of indices.
        res_for_hw = []
        for ind in indices:
            trunc_WLC = WLC[ind[0]:ind[1]]
            # Estimate scatter.
            res = est_errs(np.array([i for i in range(trunc_WLC.shape[0])]),
                           trunc_WLC)
            res_for_hw.append(np.std(res))
        reses.append(np.mean(res_for_hw))
        wlcs.append(WLC)

    if (show_plots == 2 or save_plots == 2):
        cs = [[0.75,0.75,x] for x in np.linspace(0,1,len(tested_hws))]
        for wlc,hw,c in zip(wlcs,tested_hws,cs):
            if (hw == tested_hws[0] or hw == tested_hws[-1]):
                plt.scatter(obs.exp_time, wlc, color=c, label=hw, alpha=0.75)
            else:
                plt.scatter(obs.exp_time, wlc, color=c, alpha=0.75)
        plt.legend(loc='upper left',ncols=2)
        
        if save_plots == 2:
            stagedir = os.path.join(output_dir,"stage2/plots")
            if not os.path.exists(stagedir):
                os.makedirs(stagedir)
            plt.savefig(os.path.join(stagedir,"s2_WLC{}_tested-hws-wlcs.png".format(order)),
                        dpi=300,bbox_inches='tight')
        if show_plots == 2:
            plt.show(block=True)
        plt.close()

    if (show_plots > 0 or save_plots > 0):
        plt.scatter(tested_hws, [1e6*i for i in reses])
        plt.xlabel('half-width [pixels]')
        plt.ylabel('residuals [ppm]')
        if save_plots > 0:
            stagedir = os.path.join(output_dir,"stage2/plots")
            if not os.path.exists(stagedir):
                os.makedirs(stagedir)
            plt.savefig(os.path.join(stagedir,"s2_{}_tested-hws.png".format(order)),
                        dpi=300,bbox_inches='tight')
        if show_plots > 0:
            plt.show(block=True)
        plt.close()
    
    # Find index of minimum scatter.
    ideal_halfwidth = tested_hws[reses.index(min(reses))]
    return ideal_halfwidth

def est_errs(time, flx, kick_outliers=True):
    """Simple function for estimating the scatter in the
    out-of-transit/eclipse flux.

    Args:
        time (np.array): timestamps for exposures.
        flx (np.array): oot/ooe flux.
        kick_outliers (bool, optional): Whether to remove outliers like CRs
        from the array to get more accurate scatter estimation. Defaults to True.

    Returns:
        np.array: residuals from the rampslope fit to the oot/ooe flux.
    """
    result = least_squares(residuals_,
                           np.array([1,1,0,1]),
                           args=(time,flx))
    res = residuals_(result.x,time,flx,kick_outliers)
    return res

def residuals_(fit,x,flx,kick_outliers=False):
    """Residuals function for fitting a ramp-slope time-series to oot/ooe flux.

    Args:
        fit (np.array): guess of parameters for the fit.
        x (np.array): time.
        flx (np.array): flux.
        kick_outliers (bool, optional):Whether to remove outliers like CRs
        from the array to get more accurate scatter estimation. Defaults to True.

    Returns:
         np.array: residuals of the fit.
    """
    rs = rampslope(x,fit[0],fit[1],fit[2],fit[3])
    residuals = flx-rs
    if kick_outliers:
        # Remove outlier points.
        res_mean = np.mean(residuals)
        res_sig = np.std(residuals)
        outliers = np.where(np.abs(residuals-res_mean) > 3*res_sig)[0]
        residuals = np.delete(residuals, outliers)
    return residuals

def rampslope(x,a,b,c,d):
    """A exp(b t) + c t + d trend.

    Args:
        x (np.array): time.
        a (float): amplitude of the exponential.
        b (float): "slope" of the exponential.
        c (float): linear trend slope.
        d (float): linear trend intercept.

    Returns:
        np.array: ramp-slope trend.
    """
    return a*np.exp(b*(x-np.nanmean(x))) + c*(x-np.nanmean(x)) + d