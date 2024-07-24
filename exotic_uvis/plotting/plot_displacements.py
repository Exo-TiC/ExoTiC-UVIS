import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr
import os
from exotic_uvis.plotting import plot_exposure

#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})



def plot_bkg_stars(image, exp_time, mean_loc, mean_pos, stars_pos, output_dir = None):

    """
    
    Function to plot the displacements of the background stars

    """

    plot_exposure([image], scatter_data = mean_loc, title = 'Location of background stars', 
                  stage=1, output_dir=output_dir, filename = ['bkg_stars_location'])
      
    plt.figure(figsize = (10, 7))
    plt.plot(exp_time, mean_pos[:, 0], '-o')
    plt.plot(exp_time, np.transpose(stars_pos[:, :, 0]), '-o', alpha = 0.5)
    plt.xlabel('Exposure times')
    plt.ylabel('X pixel displacement')

    stagedir = os.path.join(output_dir, 'stage1/plots/') 
    filedir = os.path.join(stagedir, 'bkg_stars_x_displacement.png')
    plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)


    plt.figure(figsize = (10, 7))
    plt.plot(exp_time, mean_pos[:, 1], '-o')
    plt.plot(exp_time, np.transpose(stars_pos[:, :, 1]), '-o', alpha = 0.5)
    plt.xlabel('Exposure times')
    plt.ylabel('Y pixel displacement')
    plt.show()

    stagedir = os.path.join(output_dir, 'stage1/plots/') 
    filedir = os.path.join(stagedir, 'bkg_stars_x_displacement.png')
    plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)

    
    return 0
