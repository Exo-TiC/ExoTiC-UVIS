from astropy.io import fits
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def full_frame_bckg_subtraction(obs, bin_number=1e5, fit='coarse', value='mode'):
    '''
    Extracts the mode or median from the full frame and subtracts this value from the image.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param bin_number: int. Number of bins used to construct the histogram.
    :param fit: str. Options are 'coarse' (use the raw histogram) or 'fine' (fit a Gaussian to the histogram). Only matters if taking the mode of the frame.
    :param value: str. Options are 'mode' (take the mode of the frame) or 'median' (take the median of the frame).
    :return: obs with sky-corrected images DataSet.
    '''
    # Track background values.
    bckgs = []

    # Ensure bin_number cannot break image.
    d_test = obs.images[0].values
    N_vals = d_test.shape[0]*d_test.shape[1]
    if bin_number > N_vals:
        print("Bin number should not exceed number of pixels to bin, reducing bin number to number of available pixels...")
        bin_number = N_vals
    
    # Iterate through frames.
    for k in range(obs.images.shape[0]):
        # Load the array and take only its finite (non-NaN) values.
        d = obs.images[k].values
        finite = d[np.isfinite(d)]

        # If you want the median, take it here.
        if value == 'median':
            bckg = np.median(finite)
            bckgs.append(bckg)
            continue
        # Build the histogram of the frame out of its finite (non-NaN) values.
        hist, bin_edges = np.histogram(finite, bins=int(bin_number))

        # If you want a coarse mode, take it here.
        if fit == 'coarse':
            ind = np.argmax(hist)
            bckg = (bin_edges[ind]+bin_edges[ind+1])/2
            bckgs.append(bckg)
            continue

        # Else, take the fit.
        elif fit == 'fine':
            # For Carlos!
            bckgs.append(bckg)
            continue

        # Build a frame to correct the data with.
        d -= bckgs[k]*np.ones_like(d)

        # Replace the obs.image with the corrected frame.
        obs.images[k] = obs.images[k].where(obs.images[k].values == d, d)
    print("All frames sky-subtracted by {} {} method.".format(fit, value))
    return obs, bckgs

def Pagul_bckg_subtraction(obs, Pagul_path, masking_parameter=0.001, median_on_columns=True):
    '''
    Scales the Pagul+ 2023 G280 sky image to each frame and subtracts the scaled image as background.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param Pagul_path: str. Path to the Pagul+ 2023 bckg image.
    :param masking_parameter: float. How aggressively to mask the source. Values of 0.001 or less recommended. A good value should make the Pagul scaling parameters similar to the frame mode.
    :param median_on_columns: bool. If True, take the median value of the Pagul+ 2023 sky image along columns. Approximately eliminates contamination from poorly-sampled parts of sky.
    :return: obs with sky-corrected images DataSet.
    '''
    # Open the Pagul+ 2023 sky image.
    with fits.open(Pagul_path) as fits_file:
        Pagul_bckg = fits_file[0].data
        if median_on_columns:
            Pagul_bckg = np.array([np.median(Pagul_bckg,axis=0),]*np.shape(Pagul_bckg)[0])
    
    # Track scaling parameters. Should be ~equal to the frame mode.
    scaling_parameters = []
    modes = []
    # Iterate through frames.
    for k in range(obs.images.shape[0]):
        # Load the array.
        d = obs.images[k].values
        x1,x2,y1,y2 = obs.subarr_coords[k].values

        # First, get the coarse frame mode and standard deviation using the frame's finite values.
        finite = d[np.isfinite(d)]
        hist, bin_edges = np.histogram(finite, bins=10**6)
        ind = np.argmax(hist)
        mode = (bin_edges[ind]+bin_edges[ind+1])/2
        sig = np.nanstd(finite)

        modes.append(mode)

        # Next, mask any sources in the frame using the frame mode and standard deviation.
        masked_frame = np.ma.masked_where(np.abs(d - mode) > masking_parameter*sig, d)
    
        # Then fit the standard bckg to the masked frame.
        def residuals_(A,x,y):
            return np.ma.sum((y - (A*x))**2)
        result = least_squares(residuals_, 1, args=(Pagul_bckg[y1:y2+1,x1:x2+1], masked_frame))
        A = result.x[0]

        # Store the scaling parameter and subtract the sky.
        scaling_parameters.append(A)
        d -= A*Pagul_bckg[y1:y2+1,x1:x2+1]

        # Replace the obs.image with the corrected frame.
        obs.images[k] = obs.images[k].where(obs.images[k].values == d, d)
    print("All frames sky-subtracted by Pagul+ 2023 method.")
    return obs, scaling_parameters, modes