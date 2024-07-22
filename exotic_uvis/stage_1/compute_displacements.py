import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import norm
from scipy import optimize
from photutils.centroids import centroid_com, centroid_2dg, centroid_quadratic
from exotic_uvis.plotting import plot_exposure


def track_bkgstars(obs, bkg_stars, window = 15, plot = False, check_all = False):

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
    if plot:
        mean_loc = list(np.mean(abs_pos, axis = 1).transpose())

        plot_exposure([image], scatter_data = mean_loc, title = 'Location of background stars')
      
        plt.figure(figsize = (10, 7))
        plt.plot(obs.exp_time.data, mean_pos[:, 0], '-o')
        plt.plot(obs.exp_time.data, np.transpose(stars_pos[:, :, 0]), '-o', alpha = 0.5)
        plt.xlabel('Exposure times')
        plt.ylabel('X pixel displacement')

        plt.figure(figsize = (10, 7))
        plt.plot(obs.exp_time.data, mean_pos[:, 1], '-o')
        plt.plot(obs.exp_time.data, np.transpose(stars_pos[:, :, 1]), '-o', alpha = 0.5)
        plt.xlabel('Exposure times')
        plt.ylabel('Y pixel displacement')
        plt.show()

    return pos


