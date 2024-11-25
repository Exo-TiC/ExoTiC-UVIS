import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import norm
from scipy import optimize
from tqdm import tqdm
from scipy import interpolate
from exotic_uvis.plotting import plot_exposure
from exotic_uvis.stage_2 import standard_extraction



def spatial_profile_smooth():
    return 0

def spatial_profile_curved():
    return 0

def spatial_profile_curved_poly():
    return 0

def window_profile(image, init_pix, fin_pix, pol_degree = 6, 
                   threshold = 6.):

    """
    
    Function to build the spatial profile of a given window
    
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




def spatial_profile(image_org, window = 40, threshold = 4., normalize = False, show_plots=0, save_plots=0):

    """"
    
    Function to calculate the spatial profile 
    
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
    if (show_plots>1) or (save_plots>1):

        plot_exposure([P_prof], title = 'Spatial_profile', min=1e-4, max=1e0,
                      show_plot=1, save_plot=0,
                      output_dir=None, filename = ['spatial_profile'])
        
        plot_exposure([image, image_org], scatter_data=[xhits, yhits], title = 'Spatial_profile', 
                      show_plot=1, save_plot=0,
                      output_dir=None, filename = ['spatial_profile'])
        
    return P_prof, image, xhits, yhits


def spatial_profile_curved_poly(sub_image_org, image, tx_main, ty_main, low_val, up_val, init_spec = None, fit_thresh = 4., fit_degree = 5, window = 50, correct_thresh = None, plot = False):


    """
    
    Function to create a spatial profile
    
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
    
    if correct_thresh:

        # mask and replace values
        mask_cr = stds_image > correct_thresh
        sub_image_org[mask_cr] = spatial_prof[mask_cr]
        xhits, yhits = np.where(mask_cr == 1)
        
    if plot:

        if correct_thresh:
            # plot difference image
            #utils.plot_image([sub_image_org, sub_image], min = 1., max = 4., show = False)
            #utils.plot_image([sub_image, spatial_prof], scatter_data = [yhits, xhits], min = 1., max = 4., show = False)

            plot_exposure([sub_image, spatial_prof], title = 'Spatial_profile', min=1e1, max=1e4,
                      show_plot=1, save_plot=0,
                      output_dir=None, filename = ['spatial_profile'])

        # compute difference image
        diff_im = (sub_image - spatial_prof)/np.sqrt(spatial_prof) 

        plt.figure(figsize = (10, 7))
        plt.imshow(diff_im, origin = 'lower', vmin = 0, vmax = 10)
        plt.colorbar()
   
        plt.figure(figsize = (10, 7))
        plt.imshow(stds_image, origin = 'lower', vmin = 0, vmax = 10)
        plt.colorbar()
        plt.show()
    


    return spatial_prof




def optimal_extraction(obs, trace_x, traces_y, width = 25, thresh = 17., prof_type = 'polyfit', iterate = False, plot = True, zero_bkg = None,
                       verbose=0, show_plots=0, save_plots=0, output_dir=None):


    """

    
    Function to extract the spectrum following the methods from Horne 1986

    
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
                                        halfwidth=12,
                                        trace_x=trace_x,
                                        trace_y=traces_y)

    # calculate median profile
    #if prof_type == 'median':
    #    prof = optimal_clean.spatial_profile_median(sub_images, plot = True) 

    # extract optimal spectrum
    for i, sub_image in enumerate(tqdm(sub_images, desc = 'Extracting Optimal Spectrum... Progress:')):

        # initialize variables and get exposure data
        opt_spec, opt_err, diff_image = [], [], []
        sub_mask = []
        err = sub_errs[i] 
     
        # initialize spectrum
        spectrum = specs[i]
        #spectrum = np.median(specs, axis = 0)

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
            prof, _, _, _ = spatial_profile(sub_image, window = 40, show_plots=show_plots, save_plots=save_plots)

        elif prof_type == 'smooth':
            prof, _, _, _ = spatial_profile_smooth(sub_image, window_len = 13, threshold = 5, plot = False)
        
        elif prof_type == 'curved_poly':
            image = images[i]
            prof = spatial_profile_curved_poly(sub_image, image, trace_x, trace_y, low_val, up_val, init_spec = None, correct_thresh=7., window = 40, plot = True)
        
        elif prof_type == 'curved_smooth':
            image = images[i]
            prof = spatial_profile_curved(sub_image, image, trace_x, trace_y, low_val, up_val, init_spec = spectrum, correct_thresh = 6., smooth_window = 7, median_window = 7, plot = True)
           
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

            if j == 450:
                print(j)
                plt.figure(figsize = (10, 7))
                plt.plot(exp_col, label='Smoothed Profile')
                plt.plot(image_col, label='Cross-dispersion profile')
                #plt.plot(spec_col*prof_col)
                plt.legend()
                plt.xlabel('Detector Pixel')
                plt.ylabel('Counts')
                plt.show(block=True)

                plt.figure()
                plt.plot(prof_col/var_col)
                plt.show(block=True)
        

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
            utils.plot_image([sub_image], scatter_data=[yhits, xhits])

        plot_profile = False
        if plot_profile:
            diff_image = np.array(diff_image).transpose()
            #diff_image = np.array(opt_spec) * prof - sub_image
            plt.figure(figsize = (10, 7))
            plt.imshow(diff_image, origin = 'lower', vmin = 0, vmax = 5)
            plt.colorbar()
            plt.show()

    if show_plots>0:
        plt.figure(figsize = (10, 7))
        plt.plot(trace_x, opt_spec)
        plt.xlabel('Pixel Detector Position')
        plt.ylabel('Extracted Counts')
        plt.title('Example of extracted spectrum')
        plt.show()


    return np.array(opt_specs), np.array(opt_specs_err)
