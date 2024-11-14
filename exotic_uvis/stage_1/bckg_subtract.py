import os
from tqdm import tqdm
import xarray as xr
from astropy.io import fits
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy.signal import medfilt, medfilt2d
import matplotlib.pyplot as plt
from exotic_uvis.plotting import plot_exposure, plot_corners, plot_bkgvals


def Pagul_bckg_subtraction(obs, Pagul_path, masking_parameter=0.001, smooth_fits=True, median_on_columns=True,
                           verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    '''
    Scales the Pagul et al. G280 sky image to each frame and subtracts the
    scaled image as background.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param Pagul_path: str. Path to the Pagul+ 2023 bckg image.
    :param masking_parameter: float. How aggressively to mask the source.
    Values of 0.001 or less recommended. A good value should make the Pagul
    scaling parameters similar to the frame mode.
    :param smooth_fits: bool. Whether to smooth the scaling parameters in time.
    Helps prevent background "flickering".
    :param median_on_columns: bool. If True, take the median value of the
    Pagul et al. sky image along columns. Approximately eliminates contamination
    from poorly-sampled parts of sky.
    :return: obs with sky-corrected images DataSet.
    '''
    # copy images
    images = obs.images.data.copy() 

    # track scaling parameters, should be ~equal to the frame mode
    scaling_parameters = []
    modes = []

    # open the Pagul et al. sky image
    with fits.open(Pagul_path) as fits_file:
        Pagul_bckg = fits_file[0].data
        if median_on_columns:
            Pagul_bckg = np.array([np.median(Pagul_bckg,axis=0),]*np.shape(Pagul_bckg)[0])

    # get the subarr_coords
    x1,x2,y1,y2 = [int(x) for x in obs.subarr_coords.values]

    # pick a bin_number that won't break image
    d_test = obs.images[0].values
    bin_number = int(0.50*d_test.shape[0]*d_test.shape[1])
    
    # iterate over all images
    for k, image in enumerate(tqdm(images, desc = 'Fitting Pagul et al. sky image... Progress:',
                                   disable=(verbose<1))):
        # first, get the coarse frame mode and standard deviation using the frame's finite values
        finite = image[np.isfinite(image)]
        hist, bin_edges = np.histogram(finite, bins=bin_number)
        ind = np.argmax(hist)
        mode = (bin_edges[ind]+bin_edges[ind+1])/2
        sig = np.nanstd(finite)

        modes.append(mode)

        # next, mask any sources in the frame using the frame mode and standard deviation
        masked_frame = np.ma.masked_where(np.abs(image - mode) > masking_parameter*sig, image)
    
        # then fit the standard bckg to the masked frame
        def residuals_(A,x,y):
            return np.ma.sum((y - (A*x))**2)
        result = least_squares(residuals_, 1, args=(Pagul_bckg[y1:y2+1,x1:x2+1], masked_frame))
        A = result.x[0]

        # store the scaling parameter
        scaling_parameters.append(A)
    scaling_parameters = np.array(scaling_parameters)
    
    if smooth_fits:
        # smooth these over
        med_A = np.median(scaling_parameters)
        sig_A = np.std(scaling_parameters)
        scaling_parameters = np.where(np.abs(scaling_parameters - med_A) > 3*sig_A,
                                      med_A, scaling_parameters)
    
    # then remove the background
    for k, image in enumerate(tqdm(images, desc = 'Removing background... Progress:',
                                   disable=(verbose<1))):
         image -= scaling_parameters[k]*Pagul_bckg[y1:y2+1,x1:x2+1]

    # save background values
    obs['bkg_vals'] = xr.DataArray(data = scaling_parameters, dims = ['exp_time'])

    # if true, plot calculated background values
    if save_plots > 0 or show_plots > 0:
        plot_bkgvals(obs.exp_time.data, scaling_parameters, method='Pagul',
                     output_dir=output_dir, save_plot=save_plots, show_plot=show_plots)
        plot_exposure([obs.images.data[1], images[1]], title = 'Background Removal Example', 
                      show_plot = (show_plots>0), save_plot = (save_plots>0), stage=1,
                      output_dir=output_dir, filename = ['before_bkg_subs', 'after_bkg_subs'])
        
    if save_plots == 2 or show_plots == 2:
        plot_bkgvals(obs.exp_time.data, scaling_parameters, method='Pagul',
                     output_dir=output_dir, save_plot=save_plots, show_plot=show_plots)
        
    obs.images.data = images
    
    return obs


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


def calculate_mode(array, hist_min, hist_max, hist_bins, exp_num = 0, 
                   fit = None, show_plots = 0, save_plots = 0, output_dir=None):
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

    # calculate hist_min and max if not given
    if (hist_min == None) or (hist_max == None): 
        hist_min, hist_max = np.percentile(array, [1, 99])
   
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

    elif fit == 'median':
        # need to mask values outside of hist_min and hist_max, then take the median
        mask = (array > hist_min) & (array < hist_max)
        bkg_val = np.median(array[mask])

    else:
        bkg_val = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1])/2
    
    # if true, plot histrogram and location of maximum
    if save_plots == 2 or show_plots == 2:
        plt.figure(figsize = (10, 7))
        plt.hist(array, bins = np.linspace(hist_min, hist_max, hist_bins), color = 'indianred', alpha = 0.7, density=False)
        plt.axvline(bkg_val, color = 'gray', linestyle = '--')

        if fit ==  'Gaussian':
            plt.plot(bin_cents, Gauss1D(bin_cents, popt[0], popt[1], popt[2], popt[3]))

        plt.axvline(np.median(array), linestyle = '--', color = 'black')
        plt.axvline((bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1])/2, linestyle = '--', color = 'blue')
        plt.xlabel('Pixel Value')
        plt.ylabel('Counts')
        plt.title(f'Background Values histogram Exposure {exp_num}')
    
        if save_plots == 2:
            stagedir = os.path.join(output_dir, f'stage1/plots/')
            if not os.path.exists(stagedir):
                os.makedirs(stagedir) 
            filedir = os.path.join(stagedir, f'bkg_histogram_exposure{exp_num}.png')
            plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)
        
        if show_plots == 2:
            plt.show(block=True)

        plt.close() # save memory

    return bkg_val


def uniform_value_bkg_subtraction(obs, fit = None, bounds = None,
                                  hist_min = -20, hist_max = 50, hist_bins = 1000,
                                  verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """ Function to compute background subtraction using image corners

    Args:
        obs (xarray): _description_
        fit (_type_, optional): _description_. Defaults to None.
        bounds (list of int, optional): bounds from which to draw the corners, if
        using corners. Use None to draw from the full frame. Defaults to None.
        hist_min (int, optional): lower bound for bckg values. Defaults to -20.
        hist_max (int, optional): upper bound for bckg values. Defaults to 50.
        hist_bins (int, optional): number of bins for calculating the mode.
        Cannot exceed number of pixels available. Defaults to 1000.
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

    # ensure bin_number cannot break image
    d_test = obs.images[0].values
    N_vals = d_test.shape[0]*d_test.shape[1]
    if hist_bins > N_vals:
        print("Bin number should not exceed number of pixels to bin, reducing bin number to half the number of available pixels...")
        hist_bins = int(0.5*N_vals)

    # iterate over all images
    for i, image in enumerate(tqdm(images, desc = 'Removing background... Progress:',
                                   disable=(verbose<1))):
        
        # calculate image background from histogram
        if bounds:
            if len(bounds) == 1:
                bound = bounds[0]
                img_bkg = calculate_mode(image[bound[0]:bound[1], bound[2]:bound[3]].flatten(), 
                                         hist_min, hist_max, hist_bins, exp_num = i, fit = fit, show_plots = show_plots,
                                           save_plots=save_plots, output_dir=output_dir)
            else:
                image_vals = []
                for bound in bounds:       
                    image_vals = np.concatenate((image_vals, image[bound[0]:bound[1], bound[2]:bound[3]].flatten()))

                img_bkg = calculate_mode(image_vals, hist_min, hist_max, hist_bins, exp_num = i, fit = fit, 
                                         show_plots = show_plots, save_plots=save_plots, output_dir=output_dir)
                
        else:
            img_bkg = calculate_mode(image.flatten(), hist_min, hist_max, hist_bins, exp_num = i, 
                                     fit = fit, show_plots = show_plots, save_plots=save_plots, output_dir=output_dir)

        # append background value
        bkg_vals.append(img_bkg)
        
        # substract background from image
        image -= img_bkg

    # save background values
    obs['bkg_vals'] = xr.DataArray(data = bkg_vals, dims = ['exp_time'])

    # if true, plot calculated background values
    if save_plots > 0 or show_plots > 0:
        method = 'full-frame'
        if bounds:
            method = 'corners'
            plot_corners(image, bounds, show_plot=(show_plots > 0), 
                         save_plot=(save_plots > 0), output_dir=output_dir)

        plot_bkgvals(obs.exp_time.data, bkg_vals, method=method,
                     output_dir=output_dir, save_plot=save_plots, show_plot=show_plots)
        plot_exposure([obs.images.data[1], images[1]], title = 'Background Removal Example', 
                      show_plot = (show_plots>0), save_plot = (save_plots>0), stage=1,
                      output_dir=output_dir, filename = ['before_bkg_subs', 'after_bkg_subs'])
        
    obs.images.data = images

    return obs

def column_by_column_subtraction(obs, rows=[i for i in range(10)], sigma=3, mask_trace=True, width=100,
                                 verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    '''
    Perform 1/f or column-by-column subtraction on each frame.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param rows: lst of int. The indices defining which rows are the background rows.
    :param sigma: float. How aggressively to remove outliers from the background region.
    :return: obs that is background-subtracted through 1/f methods.
    '''
    # copy images
    images = obs.images.data.copy() 

    # initialize background values
    bckgs = []

    # iterate over all images
    for i, image in enumerate(tqdm(images, desc = 'Removing background... Progress:',
                                   disable=(verbose<1))):
        # define background region
        bckg_region = np.ma.masked_array(image[rows,:],mask=np.zeros_like(image[rows,:]))

        # or define background as image with data masked.
        if mask_trace:
            centroids = []
            for j in range(image.shape[1]):
                centroids.append(np.argmax(image[:,j]))
            centroids = medfilt(centroids,kernel_size=7)
            mask = np.zeros_like(image)
            for j in range(image.shape[1]):
                n = 1
                if np.max(image[:,j]) > 65000:
                    n = 2 # broaden the mask near the 0th order
                l,r = (centroids[j]-n*width,centroids[j]+n*width)
                if l < 0:
                    l = 0
                if r > image.shape[1]:
                    r = image.shape[1]
                mask[l:r,j] = 1
            bckg_region = np.ma.masked_array(image,mask=mask)

            # if true, plot calculated background region
            if (save_plots > 0 or show_plots > 0) and i == 0:
                plot_exposure([bckg_region,], title = 'Col-by-col Background Region Example', 
                              show_plot = (show_plots>0), save_plot = (save_plots>0), stage=1,
                              output_dir=output_dir, filename = ['bkg_region_col-by-col',])

        # smooth outliers
        smooth_bckg = medfilt2d(bckg_region, kernel_size=7)
        std_bckg = np.ma.std(smooth_bckg)
        bckg_region = np.where(np.abs(smooth_bckg - bckg_region) > sigma*std_bckg, smooth_bckg, bckg_region)

        # median normalize on columns, kick final outliers, and append
        bckg = np.ma.median(bckg_region,axis=0)
        bckg = np.where(bckg > 100, np.ma.median(bckg),bckg)
        bckgs.append(bckg)

        # extend to full array
        bckg_region = np.array([bckg,]*np.shape(image)[0])

        # Subtract.
        image -= bckg_region
    bckgs = np.array(bckgs)

    # save background values
    obs['bkg_vals'] = xr.DataArray(data = bckgs, dims = ['exp_time','columns'])

    # if true, plot calculated background values
    if save_plots > 0 or show_plots > 0:
        plot_bkgvals(obs.exp_time.data, bckgs, method='col-by-col',
                     output_dir=output_dir, save_plot=save_plots, show_plot=show_plots)
        plot_exposure([obs.images.data[1], images[1]], title = 'Background Removal Example', 
                      show_plot = (show_plots>0), save_plot = (save_plots>0), stage=1,
                      output_dir=output_dir, filename = ['before_bkg_subs', 'after_bkg_subs'])

    obs.images.data = images

    return obs
