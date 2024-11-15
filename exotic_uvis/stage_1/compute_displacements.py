from tqdm import tqdm

import numpy as np
from photutils.centroids import centroid_com

from exotic_uvis.plotting import plot_exposure
from exotic_uvis.plotting import plot_bkg_stars, plot_0th_order


def refine_location(obs, location=None, window=20,
                    verbose=0, show_plots=0, save_plots=0, output_dir=None):
    """Function to refine the target location in the direct image

    Args:
        obs (xarray): obs.direct_image contains the direct image of the source.
        location (list, optional): initial guess for the source location.
        Defaults to None.
        window (int, optional): how far around the source to draw the window
        for centroiding. Defaults to 20.
        verbose (int, optional): how detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show.
        Defaults to 0.
        save_plots (int, optional): how many plots you want to save.
        Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.
    """

    if verbose >= 1:
        print("Refining location of source in direct image with centroiding...")
    
    # copy direct image
    image = obs.direct_image.data.copy()

     # get estimated target location
    if location:
        x_loc = location[0]
        y_loc = location[1]
    else:
        x_loc = obs.attrs['target_posx']
        y_loc = obs.attrs['target_posy']

    # if true, calculate the centroid of a window around the initial guess
    if window:
        x0, xf = int(x_loc) - window, int(x_loc) + window
        y0, yf = int(y_loc) - window, int(y_loc) + window
        
        sub_image = image[y0:yf, x0:xf]
        x1, y1 = centroid_com(sub_image)

        obs.attrs['target_posx'] = x0 + x1
        obs.attrs['target_posy'] = y0 + y1

    # otherwise, just use the default location
    else:
        obs.attrs['target_posx'] = x_loc
        obs.attrs['target_posy'] = y_loc
        

    if show_plots > 0 or save_plots > 0:
        plot_exposure([image], 
                      scatter_data = [[obs.attrs['target_posx']], [obs.attrs['target_posy']]],
                      title = 'Computed direct image target location',
                      show_plot=show_plots, save_plot=save_plots, 
                      stage=1, output_dir=output_dir, filename = [f'Target location in Direct image']) 
    
    return 


def track_bkgstars(obs, bkg_stars, window = 15,
                   verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Function to compute the x & y displacement of a given background star

    Args:
        obs (xarray): obs.images contains the images of the stars.
        bkg_stars (list of list): estimated positions of stars to track.
        window (int, optional): how far around the star to draw the window
        for centroiding. Defaults to 15.
        verbose (int, optional): how detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show.
        Defaults to 0.
        save_plots (int, optional): how many plots you want to save.
        Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        np.array, np.array: relative and average positions of each star in time.
    """

    # intialize and copy images
    stars_pos, abs_pos = [], []
    images = obs.images.data.copy()

    # iterate over all listed background stars
    for i, pos_init in tqdm(enumerate(bkg_stars),
                            desc='Tracking background stars... Progress:',
                            disable=(verbose==0)):
        
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
        
        if (show_plots==2 or save_plots==2):
            # save or show a plot of this star
            plot_exposure([images[0]], scatter_data = [x0 + x1, y0 + y1],
                          title = 'Position of star #{}'.format(i),
                          show_plot = (show_plots==2), save_plot = (save_plots==2), 
                          stage = 1, filename = ['bkg_star_no{}'.format(i)],
                          output_dir = output_dir)

        # save background star location as a function of time
        obs["star{}_disp".format(i)] = (("exp_time", "xy"), rel_pos)
        stars_pos.append(rel_pos)
        abs_pos.append(pos)

    stars_pos = np.array(stars_pos)
    mean_pos = np.mean(stars_pos, axis = 0)

    obs["meanstar_disp"] = (("exp_time", "xy"), mean_pos)
    
    # if true, plot the calculated displacements
    if (show_plots > 0) or (save_plots > 0):
        mean_loc = list(np.mean(abs_pos, axis = 1).transpose())
        plot_bkg_stars(image, obs.exp_time.data, mean_loc, mean_pos, stars_pos,
                       show_plot=(show_plots>0), save_plot=(save_plots>0),
                       output_dir=output_dir)
    
    return stars_pos, mean_pos


def track_0thOrder(obs, guess,
                   verbose=0, show_plots=0, save_plots=0, output_dir=None):
    """Tracks the 0th order through all frames using centroiding.

    Args:
        obs (xarray): obs.images contains the images.
        guess (lst of float): initial x, y position guess for the
        0th order's location.
        verbose (int, optional): how detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to show.
        Defaults to 0.
        save_plots (int, optional): how many plots you want to save.
        Defaults to 0.
        output_dir (str, optional): where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        lst of float: location of the direct image in x, y floats.
    """

    # unpack guess and integerize it
    x0, y0 = [int(i) for i in guess]

    # open lists of position
    X, Y = [], []
    for k in tqdm(range(obs.images.shape[0]),
                  desc='Tracking 0th order... Progress:',
                  disable=(verbose==0)):
        # open the kth image
        d = obs.images[k].values

        # clip a window near the guess
        window = d[y0-70:y0+70,x0-70:x0+70]

        # centroid the window
        xs, ys = centroid_com(window)

        # return to native window
        xs += x0 - 70
        ys += y0 - 70
        
        # take the source
        X.append(xs)
        Y.append(ys)
    
    if verbose > 0:
        print("Tracked 0th order in %.0f frames." % obs.images.shape[0])
    
    if (show_plots > 0 or save_plots > 0):
        plot_0th_order(obs.exp_time.data,X,Y,
                       show_plot=(show_plots>0),save_plot=(save_plots>0),output_dir=output_dir)
        
    return X, Y
