from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def standard_extraction(obs, halfwidth, trace_x, trace_y):
    '''
    Extracts a standard 1D spectrum from every image in the DataSet using the trace solution given.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param halfwidth: int. The half-width of extraction. Pulls pixels within +/- half-width vertically of the y solution.
    :param trace_x: np.array. The x positions of the pixels in the trace solutions.
    :param trace_y: np.array. The y positions of the pixels in the trace solutions.
    :return: np.array of the 1D spectra in time for the extracted order, and the truncated wavelength solutions.
    '''
    # Define traces array.
    traces = []
    err_traces = []

    # Iterate over frames.
    for k in tqdm(range(obs.images.shape[0]),desc='Extracting standard spectra... Progress:'):
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
    '''
    Short function to pull a trace region from a frame using the given solution and halfwidth.
    
    :param frame: np.array. One frame from obs.images.
    :param halfwidth: int. The half-width of extraction. Pulls pixels within +/- half-width vertically of the y solution.
    :param trace_x: np.array. The x positions of the pixels in the trace solution.
    :param trace_y: np.array. The y positions of the pixels in the trace solution.
    :return: a trace array.
    '''
    dispersion_profiles = []
    for i,x in enumerate(xs):
        x_pos = int(x)
        y_pos = range(int(ys[i])-halfwidth,int(ys[i])+halfwidth+1)
        dispersion_profile = frame[y_pos[0]:y_pos[-1],x_pos]
        dispersion_profiles.append(dispersion_profile)
    return np.array(dispersion_profiles)

def box(trace):
    """The simplest extraction method, this routine sums the trace along columns without any weighting.

    Args:
        trace (np.array): One frame in time showing the trace at integration k.

    Returns:
        np.array: 1D array of the unweighted spectrum from that trace.
    """
    return np.sum(trace,axis=1)

def determine_ideal_halfwidth(obs, trace_x, trace_y, wavs, indices=([0,10],[-10,-1])):
    '''
    Extracts multiple standard white light curves and determines the half-width that minimizes scatter out of transit/eclipse.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param trace_x: np.array. The x positions of the pixels in the trace solutions.
    :param trace_y: np.array. The y positions of the pixels in the trace solutions.
    :param wavs: np.array. The wavelength solution for each image.
    :param indices: tuple of lst of ints. Indices that define the out-of-transit/eclipse flux, for which scatter is measured.
    :return: half-width integer that minimizes scatter.
    '''
    # Initialiaze hws and residuals lists.
    tested_hws, reses = [i for i in range(5,26)], []

    # Test each half-width and measure its scatter.
    for hw in tqdm(tested_hws, desc='Testing half-widths to minimize scatter... Progress:'):
        # Get the 1D spectra.
        wavs, oneD_spec = standard_extraction(obs, hw, trace_x, trace_y)
        # Bin into a median-normalized white light curve on valid wavelength range.
        ok = (wavs[0]>2000) & (wavs[0]<8000)
        WLC = np.sum(oneD_spec[:,ok],axis=1)
        WLC /= np.median(WLC)
        plt.scatter(obs.exp_time, WLC)
        plt.savefig("WLC_hw{}.png".format(hw),dpi=300,bbox_inches='tight')
        plt.close()
        # Truncate to just the range of out-of-transit/eclipse for each set of indices.
        res_for_hw = []
        for ind in indices:
            trunc_WLC = WLC[ind[0]:ind[1]]
            # Estimate scatter.
            res = est_errs(np.array([i for i in range(trunc_WLC.shape[0])]),
                           trunc_WLC)
            res_for_hw.append(np.std(res))
        print(hw, res_for_hw)
        reses.append(np.mean(res_for_hw))

    plt.scatter(tested_hws, reses)
    plt.savefig('tested_hws.png',dpi=300,bbox_inches='tight')
    plt.close()
    print(1/0)
    
    # Find index of minimum scatter.
    ideal_halfwidth = tested_hws[reses.index(min(reses))]
    return ideal_halfwidth

def est_errs(time, flx, kick_outliers=True):
    result = least_squares(residuals_,
                           np.array([1,1,0,1]),
                           args=(time,flx))
    res = residuals_(result.x,time,flx,kick_outliers)
    return res

def residuals_(fit,x,flx,kick_outliers=False):
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
    return a*np.exp(b*x) + c*x + d