from tqdm import tqdm
import xarray as xr
from astropy.io import fits
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy.signal import medfilt2d
import matplotlib.pyplot as plt
from exotic_uvis.plotting import plot_exposure, plot_corners, plot_bkgvals


def full_frame_bckg_subtraction(obs, bin_number=1e5, fit='coarse', value='mode',
                                verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
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
        
        elif value == 'mode':
            # Build the histogram of the frame out of its finite (non-NaN) values.
            hist, bin_edges = np.histogram(finite, bins=int(bin_number))

            # If you want a coarse mode, take it here.
            if fit == 'coarse':
                ind = np.argmax(hist)
                bckg = (bin_edges[ind]+bin_edges[ind+1])/2
                bckgs.append(bckg)

            # Else, take the fit.
            elif fit == 'fine':
                # For Carlos!
                bckgs.append(bckg)

        # Build a frame to correct the data with.
        d -= bckgs[k]*np.ones_like(d)

        # Replace the obs.image with the corrected frame.
        obs.images[k] = obs.images[k].where(obs.images[k].values == d, d)
    
    if save_plots > 0 or show_plots > 0:
        plot_bkgvals(obs.exp_time.data, bckgs, output_dir=output_dir, 
                     save_plot=save_plots, show_plot=show_plots)

    print("All frames sky-subtracted by {} {} method.".format(fit, value))
    return obs, bckgs


def Pagul_bckg_subtraction(obs, Pagul_path, masking_parameter=0.001, median_on_columns=True,
                           verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
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


def Gauss1D(x, H, A, x0, sigma):
    """Function to return a 1D Gaussian profile 

    Args:
        x (_type_): _description_
        H (_type_): _description_
        A (_type_): _description_
        x0 (_type_): _description_
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """

    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def calculate_mode(array, hist_min, hist_max, hist_bins, fit = None, show_plots = 0):
    """Function to return the mode of an image

    Args:
        array (_type_): _description_
        hist_min (_type_): _description_
        hist_max (_type_): _description_
        hist_bins (_type_): _description_
        fit (_type_, optional): _description_. Defaults to None.
        show_plots (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    # create a histogram of counts 
    hist, bin_edges = np.histogram(array, bins = np.linspace(hist_min, hist_max, hist_bins))
    bin_cents = (bin_edges[:-1] + bin_edges[1:])/2

    # if true, fit gaussian to histogram and find center
    if fit == 'Gaussian':
        # fit a Gaussian profile
        popt, pcov = curve_fit(Gauss1D, 
                                bin_cents, 
                                hist, 
                                p0 = [0, np.amax(hist), bin_cents[np.argmax(hist)], (hist_max - hist_min)/4],
                                maxfev = 2000)
        
        bkg_val = popt[2]

    elif fit == 'Median':
        bkg_val = np.median(array)

    else:
        bkg_val = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1])/2
    
    # if true, plot histrogram and location of maximum
    if show_plots == 2:
        plt.figure(figsize = (10, 7))
        plt.hist(array, bins = np.linspace(hist_min, hist_max, hist_bins), color = 'indianred', alpha = 0.7, density=False)
        plt.axvline(bkg_val, color = 'gray', linestyle = '--')

        if fit ==  'Gaussian':
            plt.plot(bin_cents, Gauss1D(bin_cents, popt[0], popt[1], popt[2], popt[3]))

        plt.axvline(np.median(array), linestyle = '--', color = 'black')
        plt.axvline((bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1])/2, linestyle = '--', color = 'blue')
        plt.xlabel('Pixel Value')
        plt.ylabel('Counts')
        plt.show(block=True)

    return bkg_val


def corner_bkg_subtraction(obs, bounds = None, plot = True, check_all = False, fit = None, 
                            hist_min = -20, hist_max = 50, hist_bins = 1000,
                            verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """ Function to compute background subtraction using image corners

    Args:
        obs (_type_): _description_
        bounds (_type_, optional): _description_. Defaults to None.
        plot (bool, optional): _description_. Defaults to True.
        check_all (bool, optional): _description_. Defaults to False.
        fit (_type_, optional): _description_. Defaults to None.
        hist_min (int, optional): _description_. Defaults to -20.
        hist_max (int, optional): _description_. Defaults to 50.
        hist_bins (int, optional): _description_. Defaults to 1000.
        verbose (int, optional): _description_. Defaults to 0.
        show_plots (int, optional): _description_. Defaults to 0.
        save_plots (int, optional): _description_. Defaults to 0.
        output_dir (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # copy images
    images = obs.images.data.copy() 

    # initialize background values
    bkg_vals = []

    # iterate over all images
    for i, image in enumerate(tqdm(images, desc = 'Removing background... Progress:')):
        
        # calculate image background from histogram
        if bounds:
            if len(bounds) == 1:
                bound = bounds[0]
                img_bkg = calculate_mode(image[bound[0]:bound[1], bound[2]:bound[3]].flatten(), 
                                         hist_min, hist_max, hist_bins, fit = fit, show_plots = show_plots)
            else:
                image_vals = []

                for bound in bounds:       
                    image_vals = np.concatenate((image_vals, image[bound[0]:bound[1], bound[2]:bound[3]].flatten()))

                img_bkg = calculate_mode(image_vals, hist_min, hist_max, hist_bins, fit = fit, show_plots = show_plots)
                
        else:
            img_bkg = calculate_mode(image.flatten(), hist_min, hist_max, hist_bins, fit = fit, show_plots = show_plots)

        # append background value
        bkg_vals.append(img_bkg)
        
        # substract background from image
        image -= img_bkg

    # save background values
    obs['bkg_vals'] = xr.DataArray(data = bkg_vals, dims = ['exp_time'])

    # if true, plot calculated background values
    if save_plots > 0 or show_plots > 0:
        #plot_corners([images[0]], bounds, output_dir=output_dir)
        plot_bkgvals(obs.exp_time.data, bkg_vals, output_dir=output_dir, 
                     save_plot=save_plots, show_plot=show_plots)
        plot_exposure([obs.images.data[1], images[1]], title = 'Background Removal Example', stage=1,
                      output_dir=output_dir, filename = ['before_bkg_subs', 'after_bkg_subs'])

    obs.images.data = images

    return obs


def column_by_column_subtraction(obs, rows=[i for i in range(10)], sigma=3):
    '''
    Perform 1/f or column-by-column subtraction on each frame.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param rows: lst of int. The indices defining which rows are the background rows.
    :param sigma: float. How aggressively to remove outliers from the background region.
    :return: obs that is background-subtracted through 1/f methods.
    '''
    # Iterate on frames.
    for k in tqdm(range(obs.images.shape[0]), desc = 'Removing background... Progress:'):
        # Get data.
        d = obs.images[k].values

        # Define background region.
        bckg_region = d[rows,:]

        # Smooth outliers.
        smooth_bckg = medfilt2d(bckg_region, kernel_size=3)
        std_bckg = np.std(smooth_bckg)
        bckg_region = np.where(np.abs(smooth_bckg - bckg_region) > sigma*std_bckg, smooth_bckg, bckg_region)

        # Median normalize on columns and extend to full array.
        bckg_region = np.array([np.median(bckg_region,axis=0),]*np.shape(d)[0])

        # Subtract.
        d -= bckg_region

        # Replace the obs.image with the corrected frame.
        obs.images[k] = obs.images[k].where(obs.images[k].values == d, d)
    print("All frames sky-subtracted by column-by-column method method.")
    return obs
