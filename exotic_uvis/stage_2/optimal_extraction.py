import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import medfilt

from exotic_uvis.plotting import plot_exposure
from exotic_uvis.stage_2 import standard_extraction


def spatial_profile_curved():
    return 0


def spatial_profile_smooth(image_org, kernel = 11, threshold = 5., std_window = 20, 
                           median_window = 7, show_plots=0, save_plots=0, output_dir=0):
    """Builds a spatial profile using 1D smoothing.

    Args:
        image_org (array-like): original images to be modelled and cleaned.
        kernel (int, optional): odd int which defines the size of the row filter.
        Defaults to 11.
        threshold (float, optional): threshold at which to kick outliers.
        Defaults to 5..
        std_window (int, optional): window over which to calculate the
        standard deviation of the row. Defaults to 20.
        median_window (int, optional): window over which to calculate
        the median of the row. Defaults to 7.
        show_plots (int, optional): how many plots you want to show.
        Defaults to 0.
        save_plots (int, optional): how many plots you want to save.
        Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        array-like, array-like, array-like, array-like: the spatial profile,
        cleaned image, and maps of where pixels were hit in x and y.
    """
    
    # copy image and initialize
    image = image_org.copy()
    nrows, ncols = np.shape(image)
    P_prof, xhits, yhits = [], [], []

    # iterate over all rows
    for j in range(nrows):
        
        # get row
        row = image[j].copy()
        row_mask = ~np.isnan(row)
        badpixels = True

        while badpixels:
            # replace masked values with median (otherwise the window might fail)
            for ind in np.where(row_mask == 0)[0]:
                row[ind] = np.median(row[np.amax((0, ind - median_window)):ind+median_window])
               
            # smooth row with median filter
            row_model = medfilt(row, kernel)

            # compute residuals
            res = np.ma.array(row - row_model, mask = ~row_mask)

            # calculate standard deviation with a window too 
            #row_std = np.ma.std(res)
            row_std = np.zeros_like(row)
            for i, row_val in enumerate(row):
                row_std[i] = np.std(res[np.amax((0, i-std_window)):i+std_window])
                
            #dev_row = np.ma.abs(res) / np.ma.std(res)
            dev_row = np.ma.abs(res)/row_std
            max_dev_ind = np.ma.argmax(dev_row)

            # if maximum deviation pixel above threshold, mask pixel
            if dev_row[max_dev_ind] > threshold:
                row_mask[max_dev_ind] = False

            # if no more pixels above threshold, finish
            else: 
                badpixels = False
                P_prof.append(row_model)

                # replace bad pixels with row model
                for ind, good_pix in enumerate(row_mask):
                    if not good_pix:         
                        image[j, ind] = row_model[ind]
                        xhits.append(ind)
                        yhits.append(j)

        # just some inside plots for sanity check
        if (show_plots == 2 or save_plots == 2):   
            plt.figure()
            #plt.plot(image_org[j]/f_init)
            plt.plot(image[j])
            plt.plot(row_model)

            if save_plots == 2:
                plot_dir = os.path.join(output_dir,'plots')
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(os.path.join(plot_dir, 'spatialprofile_smooth.png'),
                            dpi=300,bbox_inches='tight')
            
            if show_plots == 2:
                plt.show(block=True)
            
            plt.close() # save memory
    
    # normalize spatial profile
    P_prof = np.array(P_prof)
    P_prof[P_prof < 0.] = 0
    P_prof = P_prof / np.sum(P_prof, axis = 0)
     
    return P_prof, image, np.array(xhits), np.array(yhits)


def spatial_profile_median(images, show_plots=0, save_plots=0, output_dir=None):
    """Uses the entire time series of data to compute a median image,
    normalized and then used as the spatial profile.

    Args:
        images (array-like): full time series of observation.
        show_plots (int, optional): how many plots you want to show.
        Defaults to 0.
        save_plots (int, optional): how many plots you want to save.
        Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        array-like: spatial profile for optimal extraction.
    """

    # calculate median image
    P_prof = np.median(images, axis = 0)

    # set negative values to 0
    P_prof[P_prof < 0] = 0

    # normalize profile along the cross_dispersion direction
    P_prof = P_prof / np.sum(P_prof, axis = 0)

    # if true, plot the computed spatial profile
    if (show_plots==2) or (save_plots==2):
        plot_exposure([P_prof], title = 'Example of Spatial profile', min=1e-4, max=1e0,
                      show_plot=(show_plots==2), save_plot=(save_plots==2),
                      output_dir=output_dir, filename = [f'spatialprofile_median'])
        
    return P_prof


def window_profile(image, init_pix, fin_pix, pol_degree = 6, 
                   threshold = 6.):
    """Uses row-wise polynomials to build 

    Args:
        image (_type_): _description_
        init_pix (_type_): _description_
        fin_pix (_type_): _description_
        pol_degree (int, optional): _description_. Defaults to 6.
        threshold (_type_, optional): _description_. Defaults to 6..

    Returns:
        array-like, array-like, array-like, array-like: _description_
    """

    # initialize
    P_win, xhit, yhit, stds = [], [], [], []

    D_win = image[:, init_pix : fin_pix]
    col_pix = np.arange(D_win.shape[1])

    # iterate over all rows
    for i, row in enumerate(D_win):
        
        D_row = row.copy()
        row_mask = ~np.isnan(D_row)
        badpixels = True

        # loop while bad pixels are found
        while badpixels:

            # fit a polynomial to the row
            coeffs = np.polyfit(col_pix[row_mask], D_row[row_mask], deg = pol_degree)
            row_fit = np.polyval(coeffs, col_pix)

            # create residuals arrays and calculation deviation
            res = np.ma.array(D_row - row_fit, mask = ~row_mask)
            dev_row = res / np.ma.std(res)
            #dev_row = np.ma.abs(res) / np.ma.std(res)

            # location max deviation pixel
            max_dev_ind = np.ma.argmax(dev_row)

            # if maximum deviation pixel above threshold, mask pixel
            if dev_row[max_dev_ind] > threshold:
                row_mask[max_dev_ind] = False

            # if no more pixels above threshold, finish
            else: 
                P_win.append(row_fit)

                # replace bad pixels with row model
                for ind, good_pix in enumerate(row_mask):
                    if not good_pix:         
                        image[i, init_pix + ind] = row_fit[ind]
                        xhit.append(init_pix + ind)
                        yhit.append(i)
                        
                badpixels = False

        stds.append(np.abs(D_row - row_fit)/ np.ma.std(res))

    P_win = np.array(P_win)
    P_win[P_win < 0.] = 0

    return P_win, xhit, yhit, stds


def spatial_profile(exp_ind, image_org, window = 40, threshold = 4., normalize = False,
                    show_plots=0, save_plots=0, output_dir=0):
    """Builds a spatial profile using row-wise polynomial fits for every
    row in the image. Used in the "polyfit" method of profile building.

    Args:
        exp_ind (int): index of the exposure.
        image_org (array-like): original image to be modelled.
        window (int, optional): length of the window. Defaults to 40.
        threshold (_type_, optional): _description_. Defaults to 4..
        normalize (bool, optional): _description_. Defaults to False.
        show_plots (int, optional): _description_. Defaults to 0.
        save_plots (int, optional): _description_. Defaults to 0.
        output_dir (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    # copy image and initialize
    image = image_org.copy()
    n_cols = np.shape(image)[1]

    # normalize
    if normalize:
        image /= np.median(image, axis = 0)

    # initialize
    P_prof = np.zeros_like(image)
    xhits, yhits = [], []

    # iterate over all windows
    for init_pix in range(0, n_cols, window):
        
        fin_pix = min(init_pix + window, n_cols)
        if fin_pix == n_cols: 
            init_pix = fin_pix - window

        # create window profile
        P_win, xhit, yhit, _ = window_profile(image, init_pix, fin_pix, threshold = threshold)

        # normalize
        P_win = P_win / np.sum(P_win, axis = 0)
        P_prof[:, init_pix : fin_pix] = P_win

        # save corrected pixel locations
        xhits = np.concatenate((xhits, xhit))
        yhits = np.concatenate((yhits, yhit))
    
    # if true, plot spatial profile
    if (show_plots==2) or (save_plots==2):

        #plot_exposure([image_org], title = 'Spatial_profile', min=1e1, max=1e4,
        #              show_plot=1, save_plot=0,
        #              output_dir=None, filename = ['spatial_profile'])

        plot_exposure([P_prof], title = f'Example of Spatial profile Exposure {exp_ind}', min=1e-4, max=1e0,
                      show_plot=(show_plots==2), save_plot=(save_plots==2),
                      output_dir=output_dir, filename = [f'spatial_profile_exp{exp_ind}'])
        
        #plot_exposure([image, image_org], scatter_data=[xhits, yhits], title = 'Spatial_profile', 
        #              show_plot=1, save_plot=0,
        #              output_dir=None, filename = ['spatial_profile'])
        
    return P_prof, image, xhits, yhits


def spatial_profile_curved_poly(exp_ind, sub_image_org, image, tx_main, ty_main, low_val, up_val, init_spec = None, 
                                fit_thresh = 4., fit_degree = 5, window = 50, correct_thresh = None,
                                show_plots=0, save_plots=0, output_dir=0):
    """Builds a spatial profile using curved polynomial fits.
    Used in the "curved_poly" method of profile building.

    Args:
        exp_ind (int): index of the exposure.
        sub_image_org (array-like): original image to fit a profile to.
        image (_type_): _description_
        tx_main (_type_): _description_
        ty_main (_type_): _description_
        low_val (_type_): _description_
        up_val (_type_): _description_
        init_spec (_type_, optional): _description_. Defaults to None.
        fit_thresh (_type_, optional): _description_. Defaults to 4..
        fit_degree (int, optional): _description_. Defaults to 5.
        window (int, optional): _description_. Defaults to 50.
        correct_thresh (_type_, optional): _description_. Defaults to None.
        show_plots (int, optional): how many plots you want to show.
        Defaults to 0.
        save_plots (int, optional): how many plots you want to save.
        Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        _type_: _description_
    """

    # copy image data and extract y values
    sub_image = sub_image_org.copy()
    y_data = range(np.shape(image)[0])
    y_vals = range(low_val, up_val)

    # calculate displacements
    margin = 2
    down_disp = low_val - np.amax(ty_main) - margin
    up_disp = up_val - np.amin(ty_main) + margin
    
    # define displacements
    ndisps = 500
    disps = np.linspace(down_disp, up_disp, ndisps)

    # compute grid
    curve_image = []
    xcoord_grid, ycoord_grid, val_grid = [], [], []
    
    # check if initial spectrum is given
    if init_spec is None:
        init_spec = np.ones_like(tx_main)

    # iterate over all pixels
    for j, pix in enumerate(tx_main):

        prof = image[:, int(pix)] / init_spec[j]
        prof_fit = interpolate.interp1d(y_data, prof, kind = 'cubic')

        curve_image.append(prof_fit(ty_main[j] + disps))

    curve_image = np.array(curve_image).transpose()

    # initialize
    n_cols = np.shape(curve_image)[1]
    P_prof = np.zeros_like(curve_image)
    stds = np.zeros_like(curve_image)
    xhits, yhits = [], []

    # iterate over all windows
    for init_pix in range(0, n_cols, window):
        
        fin_pix = min(init_pix + window, n_cols)
        if fin_pix == n_cols: 
            init_pix = fin_pix - window

        # create window profile
        curve_image_copy = curve_image.copy()
        P_win, xhit, yhit, stds_win = window_profile(curve_image_copy, init_pix, fin_pix, threshold = fit_thresh, pol_degree = fit_degree) # curve image is being modified here

        # normalize
        P_prof[:, init_pix:fin_pix] = P_win

        # compute stds
        stds[:, init_pix:fin_pix] = stds_win

        # save corrected pixel locations
        xhits = np.concatenate((xhits, xhit))
        yhits = np.concatenate((yhits, yhit))
    
    # compute grid of values
    val_grid = P_prof * init_spec 

    # compute x and y coordinates grid
    ycoord_grid = np.ones_like(curve_image)*ty_main + np.transpose(np.transpose(np.ones_like(curve_image))*disps)
    xcoord_grid = np.ones_like(curve_image)*tx_main 

    # compute grid points
    grid_points = np.stack((np.ravel(xcoord_grid), np.ravel(ycoord_grid)))
    grid_x, grid_y = np.meshgrid(tx_main, y_vals)
    
    # compute spatial profile by interpolating in grid points
    spatial_prof = interpolate.griddata(np.transpose(grid_points), np.ravel(val_grid), (np.ravel(grid_x), np.ravel(grid_y)), method = 'linear')
    spatial_prof = np.reshape(spatial_prof, (len(y_vals), len(tx_main)))
    spatial_prof[spatial_prof < 0] = 0

    # compute standard deviations
    stds_image = interpolate.griddata(np.transpose(grid_points), np.ravel(stds), (np.ravel(grid_x), np.ravel(grid_y)), method = 'linear')
    stds_image = np.reshape(stds_image, (len(y_vals), len(tx_main)))
    
    #if correct_thresh:
    # mask and replace values
    mask_cr = stds_image > correct_thresh
    sub_image_org[mask_cr] = spatial_prof[mask_cr]
    xhits, yhits = np.where(mask_cr == 1)

    # normalize profile along the cross_dispersion direction
    spatial_prof = spatial_prof/np.sum(spatial_prof, axis = 0)
        
    if show_plots==2 or save_plots==2:

        #if correct_thresh:
        # plot difference image
        #utils.plot_image([sub_image_org, sub_image], min = 1., max = 4., show = False)
        #utils.plot_image([sub_image, spatial_prof], scatter_data = [yhits, xhits], min = 1., max = 4., show = False)

        plot_exposure([spatial_prof], scatter_data = [yhits, xhits], #[sub_image, spatial_prof]
                        title = f'Example of Spatial profile Exposure {exp_ind}', min=1e-4, max=1e0,
                        show_plot=(show_plots==2), save_plot=(save_plots==2),
                        output_dir=output_dir, filename = [f'spatial_profile_exp{exp_ind}'])

        # compute difference image
        #diff_im = (sub_image - spatial_prof)/np.sqrt(spatial_prof) 

        #plt.figure(figsize = (10, 7))
        #plt.imshow(diff_im, origin = 'lower', vmin = 0, vmax = 10)
        #plt.colorbar()
   
        #plt.figure(figsize = (10, 7))
        #plt.imshow(stds_image, origin = 'lower', vmin = 0, vmax = 10)
        #plt.colorbar()
        #plt.show()

    return spatial_prof


def optimal_extraction(obs, trace_x, traces_y, width = 25, thresh = 17., prof_type = 'polyfit', 
                       iterate = False, zero_bkg = None,
                       verbose=0, show_plots=0, save_plots=0, output_dir=None):
    """Performs an optimal extraction with a spatial profile of choice following
    the methods of Horne 1986.

    Args:
        obs (xarray): dataset from which we will extract the 1D spectra.
        trace_x (array-like): x column solutions of the trace to extract.
        traces_y (arary-like): y row solutions of the trace to extract.
        width (int, optional): aperture halfwidth for extraction. For optimal,
        ideally use a very large window since the weighting will take care of
        the rest. Defaults to 25.
        thresh (float, optional): _description_. Defaults to 17..
        prof_type (str, optional): the type of profile to use for optimal
        extraction. Options are 'median', 'polyfit', 'smooth', 'curved_poly',
        or 'curved_smooth'. Defaults to 'polyfit'.
        iterate (bool, optional): _description_. Defaults to False.
        zero_bkg (_type_, optional): _description_. Defaults to None.
        verbose (int, optional): how detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show.
        Defaults to 0.
        save_plots (int, optional): how many plots you want to save.
        Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        array-like, array-like: optimally-extracted 1D spectra and uncertainties.
    """

    #initialize
    opt_specs, opt_specs_err = [], []
    images = obs.images.data.copy()
    errors = obs.errors.data.copy()

    # Define subarray for extraction
    margin = 5
    low_val, up_val = int(np.amin(traces_y)) - width - margin, int(np.amax(traces_y)) + width + margin

    # crop images and error
    sub_images = images[:, low_val : up_val, int(trace_x[0]) : int(trace_x[-1]) + 1]
    sub_errs = errors[:, low_val : up_val, int(trace_x[0]) : int(trace_x[-1]) + 1]

    # get initial spectrum
    specs, specs_err = standard_extraction(obs,
                                           halfwidth=12, # why not use the width input into this function?
                                           trace_x=trace_x,
                                           trace_y=traces_y)

    # calculate median profile
    if prof_type == 'median':
        prof = spatial_profile_median(sub_images, show_plots=show_plots, 
                                      save_plots=save_plots, output_dir=output_dir) 
        
    # generate random number for plotting
    plot_ind = np.random.randint(0, np.shape(sub_images)[0])

    # extract optimal spectrum
    for i, sub_image in enumerate(tqdm(sub_images, desc = 'Extracting optimal spectrum... Progress')):

        # initialize variables and get exposure data
        opt_spec, opt_err, diff_image = [], [], []
        sub_mask = []
        err = sub_errs[i] 
     
        # initialize spectrum
        spectrum = specs[i]

        # calculate relative trace
        if traces_y.ndim == 2:
            trace_y = traces_y[i]
        else:
            trace_y = traces_y
        tracey_rel = trace_y - low_val

        # define hit image
        hit_image = np.zeros_like(sub_image)

        # calculate profile in region of interest:
        if prof_type == 'polyfit':
            prof, _, _, _ = spatial_profile(i, sub_image, window = 40, show_plots = show_plots,
                                           save_plots=save_plots, output_dir=output_dir)

        elif prof_type == 'smooth':
            prof, _, _, _ = spatial_profile_smooth(i, sub_image, window_len = 13, threshold = 5,
                                                   show_plots=show_plots, save_plots=save_plots, output_dir=output_dir)
        
        elif prof_type == 'curved_poly':
            image = images[i]
            prof = spatial_profile_curved_poly(i, sub_image, image, trace_x, trace_y, low_val, up_val, init_spec = None, correct_thresh=7., window = 40,
                                               show_plots = show_plots, save_plots=save_plots, output_dir=output_dir)
        
        elif prof_type == 'curved_smooth':
            image = images[i]
            prof = spatial_profile_curved(i, sub_image, image, trace_x, trace_y, low_val, up_val, init_spec = spectrum, correct_thresh = 6., smooth_window = 7, median_window = 7,
                                          show_plots = show_plots, save_plots=save_plots, output_dir=output_dir)
        
        if (show_plots==1 or save_plots==1) and (i == 0):
            plot_exposure([prof], title = f'Example of Spatial profile Exposure {i}', min=1e-4, max=1e0,
                    show_plot=(show_plots > 0), save_plot=(save_plots > 0),
                    output_dir=output_dir, filename = [f'spatial_profile_exp{i}'])

           
        # iterate over each column
        for j, pix in enumerate(trace_x):
            
            # get trace value
            trace_rel = tracey_rel[j]

            # calculate lower and upper limits
            low_lim = trace_rel - width
            up_lim = trace_rel + width

            # get aperture arrays
            prof_col = prof[int(np.ceil(low_lim)) : int(np.floor(up_lim)) + 1, j]
            image_col = sub_image[int(np.ceil(low_lim)) : int(np.floor(up_lim)) + 1, j]
            var_col = err[int(np.ceil(low_lim)) : int(np.floor(up_lim)) + 1, j]**2
            
            # get initial spectrum and variance estimates
            spec_col = spectrum[j]
            prof_col /= np.sum(prof_col)

            # define readout variance 
            read_var = obs.read_noise.data[i]
            #print(read_var)

            # define background, if true, add background from zeroth order flux
            if zero_bkg is None:
                bkg = obs.bkg_vals.data[i]
            else: 
                bkg = zero_bkg[i, int(np.ceil(low_lim + low_val)) : int(np.floor(up_lim + low_val)) + 1, int(pix)] + obs.bkg_vals.data[i]
        
            # calculate expected spectrum
            exp_col = spec_col * prof_col

            # compute variance
            var_col = np.abs(exp_col + bkg) + read_var

            # compute spectrum
            spec_col = np.sum(prof_col*image_col/var_col) / np.sum(prof_col*prof_col / var_col)
            diff_image.append(np.abs(image_col - exp_col)/np.sqrt(exp_col))

            # define column mask
            badpixels = False
            mask_col = np.ones_like(prof_col, dtype = 'bool')

            if iterate:
                badpixels = True

            # loop while outliers are still found
            while badpixels: 

                # mask cosmic rays
                stds = np.abs(image_col - exp_col)*mask_col / np.sqrt(var_col)
                loc = np.argmax(stds)

                if stds[loc] > thresh:
                    mask_col[loc] = False
                else:
                    badpixels = False
                 
                # calculte expected spectrum
                exp_col = spec_col * prof_col
        
                # compute variance
                var_col = np.abs(exp_col + bkg) + read_var

                # compute spectrum
                spec_col = np.sum(prof_col*image_col*mask_col / var_col) / np.sum(prof_col*prof_col*mask_col / var_col)

            # extract spectrum
            opt_spec.append(spec_col)
            opt_err.append(np.sqrt(np.sum(prof_col*mask_col) / np.sum(prof_col*prof_col*mask_col / var_col)))
           
            # append corrected pixels in hit image
            hit_image[int(np.ceil(low_lim)) : int(np.floor(up_lim)) + 1, j] = ~mask_col
        
        # append results
        opt_specs.append(opt_spec)
        opt_specs_err.append(opt_err)
        
        # plot masked pixels
        if iterate:
            xhits, yhits = np.where(hit_image == 1)
            #utils.plot_image([sub_image], scatter_data=[yhits, xhits])

    return np.array(opt_specs), np.array(opt_specs_err)
