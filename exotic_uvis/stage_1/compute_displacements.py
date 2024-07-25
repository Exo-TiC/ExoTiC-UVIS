import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import norm
from scipy import optimize
from photutils.centroids import centroid_com, centroid_2dg, centroid_quadratic
from exotic_uvis.plotting import plot_exposure
from exotic_uvis.plotting import plot_bkg_stars

def track_bkgstars(obs, bkg_stars, window = 15, verbose_plots = 0, check_all = False, output_dir = None):

    """
    
    Function to compute the x & y displacement of a given background star
    
    """

    # intialize and copy images
    stars_pos, abs_pos = [], []
    images = obs.images.data.copy()

    # iterate over all listed background stars
    for i, pos_init in enumerate(bkg_stars):
        
        # initialize position
        pos = []

        # get window limits
        x0, xf = pos_init[0] - window, pos_init[0] + window
        y0, yf = pos_init[1] - window, pos_init[1] + window

        # iterate over all images
        for image in images:

            # define region around background star
            sub_image = image[y0:yf, x0:xf]
            
            # compute centroid
            x1, y1 = centroid_com(sub_image)

            # append location
            pos.append([x0 + x1, y0 + y1])
        
        rel_pos = np.array(pos) - pos[0]
        
        if check_all:
            plot_exposure([images[0]], scatter_data = [x0 + x1, y0 + y1])

        # save background star location as a function of time
        obs["star{}_disp".format(i)] = (("exp_time", "xy"), rel_pos)
        stars_pos.append(rel_pos)
        abs_pos.append(pos)

    stars_pos = np.array(stars_pos)
    mean_pos = np.mean(stars_pos, axis = 0)

    obs["meanstar_disp"] = (("exp_time", "xy"), mean_pos)
    
    # if true, plot the calculated displacements
    if verbose_plots > 0:
        mean_loc = list(np.mean(abs_pos, axis = 1).transpose())
        plot_bkg_stars(image, obs.exp_time.data, mean_loc, mean_pos, stars_pos, output_dir=output_dir)

    
    return pos

def track_0thOrder(obs, guess):
    '''
    Tracks the 0th order through all frames using centroiding.
    
    :param obs: xarray. Its obs.images DataSet contains the images.
    :param guess: lst of float. Initial x, y position guess for the 0th order's location.
    :return: location of the direct image in x, y floats.
    '''    
    # Correct direct image guess to be appropriate for spec images.
    # FIX: hardcoded based on WASP-31 test. There must be a better way...
    guess[0] += 100
    guess[1] += 150

    # Open lists of position.
    X, Y = [], []
    for k in range(obs.images.shape[0]):
        # Open the kth image.
        d = obs.images[k].values

        # Unpack guess and integerize it.
        x0, y0 = [int(i) for i in guess]

        # Clip a window near the guess.
        window = d[y0-70:y0+70,x0-70:x0+70]

        # Centroid the window.
        xs, ys = centroid_com(window)
        print(xs, ys)

        # Return to native window.
        xs += x0 - 70
        ys += y0 - 70
        
        # Take the source.
        X.append(xs)
        Y.append(ys)
    print("Tracked 0th order in %.0f frames." % obs.images.shape[0])
    return X, Y
