import numpy as np

def standard_extraction(obs, halfwidth, trace_x, trace_y, wavs):
    '''
    Extracts a standard 1D spectrum from every image in the DataSet using the trace solution given.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param halfwidth: int. The half-width of extraction. Pulls pixels within +/- half-width vertically of the y solution.
    :param trace_x: np.array. The x positions of the pixels in the trace solution.
    :param trace_y: np.array. The y positions of the pixels in the trace solution.
    :param wavs: np.array. The wavelength solution for each image.
    :return: np.array of the 1D spectra in time for the extracted order, and the truncated wavelength solutions.
    '''
    # Define traces array.
    traces = []
    # Iterate over frames.
    for k in range(obs.images.shape[0]):
        # Get array values of the current frame.
        frame = obs.images[k].values

        # Get trace solution for this frame.
        xs = trace_x[k]
        ys = trace_y[k]

        # Pull out just the trace from the frame.
        dispersion_profiles = []
        for i,x in enumerate(xs):
            x_pos = int(x)
            y_pos = range(int(ys[i])-halfwidth,int(ys[i])+halfwidth+1)
            dispersion_profile = frame[x_pos,y_pos[0]:y_pos[-1]]
            dispersion_profiles.append(dispersion_profile)
        trace = np.array(dispersion_profiles)
        traces.append(trace)
        
    # Numpy array it.
    traces = np.array(traces)

    # Store the 1D spectra, and prepare to update the wavelengths with the truncated 2000 <= w <= 8000 solution.
    wavs_new = []
    oneD_spec = []

    # Extract 1D spectrum using the standard method.
    for k in range(traces.shape[0]):
        #show_frame(trace[:,:,k],"Frame {} from which spectrum is extracted".format(k))
        trunc_wavs, flx = box(trace[:,:,k], wavs[k])
        wavs_new.append(trunc_wavs)
        oneD_spec.append(flx)
    
    return np.array(trunc_wavs), np.array(oneD_spec)

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