import numpy as np
from scipy.optimize import least_squares

def standard_extraction(obs, halfwidth, trace_x, trace_y, wavs):
    '''
    Extracts a standard 1D spectrum from every image in the DataSet using the trace solution given.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param halfwidth: int. The half-width of extraction. Pulls pixels within +/- half-width vertically of the y solution.
    :param trace_x: np.array. The x positions of the pixels in the trace solutions.
    :param trace_y: np.array. The y positions of the pixels in the trace solutions.
    :param wavs: np.array. The wavelength solution for each image.
    :return: np.array of the 1D spectra in time for the extracted order, and the truncated wavelength solutions.
    '''
    # Define traces array.
    traces = []
    err_traces = []

    # Iterate over frames.
    for k in range(obs.images.shape[0]):
        # Get array values and error values of the current frame.
        frame = obs.images[k].values
        err = obs.errors[k].values

        # Get trace solution for this frame.
        xs = trace_x[k]
        ys = trace_y[k]

        # Pull out just the trace from the frame.
        trace = get_trace(frame, halfwidth, xs, ys)
        err_trace = get_trace(err, halfwidth, xs, ys)

        traces.append(trace)
        err_traces.append(err_trace)
        
    # Numpy array it.
    traces = np.array(traces)
    err_traces = np.array(err_traces)

    # Store the 1D spectra and errors, and prepare to update the wavelengths with the truncated 2000 <= w <= 8000 solution.
    wavs_new = []
    oneD_spec = []
    spec_err = []

    # Extract 1D spectrum using the standard method.
    for k in range(traces.shape[0]):
        #show_frame(trace[:,:,k],"Frame {} from which spectrum is extracted".format(k))
        trunc_wavs, err = box(err_traces[:,:,k], wavs[k])
        trunc_wavs, flx = box(traces[:,:,k], wavs[k])
        wavs_new.append(trunc_wavs)
        oneD_spec.append(flx)
        spec_err.append(err)
    
    return np.array(trunc_wavs), np.array(oneD_spec), np.array(spec_err)

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
        dispersion_profile = frame[x_pos,y_pos[0]:y_pos[-1]]
        dispersion_profiles.append(dispersion_profile)
    return np.array(dispersion_profiles)

def box(trace, wavs):
    '''
    The simplest extraction method, this routine sums the trace along columns without any weighting.

    :param trace: 2D array. One frame in time of the trace array of data.
    :param wavs: 1D array. Wavelength solution on the x axis.
    :return: the unweighted spectrum for this trace.
    '''
    ysum = np.sum(trace,axis=0)
    ok = (wavs>2000) & (wavs<8000)
    flx = ysum[ok]
    #flx = np.where(flx == np.inf, 0, flx)
    return wavs[ok], flx

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
    tested_hws = [i for i in range(5,26)]
    reses = []

    # Test each half-width and measure its scatter.
    for hw in tested_hws:
        # Get the 1D spectra.
        wavs, oneD_spec = standard_extraction(obs, hw, trace_x, trace_y, wavs)
        # Bin into a median-normalized white light curve.
        WLC = np.sum(oneD_spec,axis=1)
        WLC /= np.median(WLC)
        # Estimate scatter.
        res = est_errs([i for i in range(WLC.shape[0])],
                       WLC)
        reses.append(np.std(res))
    
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