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


def plot_corners(image, corners, output_dir = None):
    """Function to plot exposure with rectangles to indicate the corners used for background subtraction

    Args:
        image (_type_): _description_
        corners (_type_): _description_
        output_dir (_type_, optional): _description_. Defaults to None.
    """

    plot_exposure(image, show = False)
    ax = plt.gca()

    for corner in corners:
        rect = patches.Rectangle((corner[2], corner[0]), corner[3] - corner[2], 
                                 corner[1] - corner[0], linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)
    #plt.show()

    stagedir = os.path.join(output_dir, 'stage1/plots/') 
    filedir = os.path.join(stagedir, 'bkg_corners.png')
    plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)


    return 


def plot_bkgvals(exp_times, bkg_vals, output_dir = None, save_plot = False, show_plot = False):
    """Function to create background subtraction plots

    Args:
        exp_times (_type_): _description_
        bkg_vals (_type_): _description_
        output_dir (_type_, optional): _description_. Defaults to None.
        save_plot (bool, optional): _description_. Defaults to False.
        show_plot (bool, optional): _description_. Defaults to False.
    """

    plt.figure(figsize = (10, 7))
    plt.plot(exp_times, bkg_vals, '-o')
    plt.xlabel('Exposure')
    plt.ylabel('Background Counts')
    plt.title('Image background per exposure')

    stagedir = os.path.join(output_dir, 'stage1/plots/') 
    filedir = os.path.join(stagedir, 'bkg_values.png')
    
    if save_plot:
        plt.savefig(filedir, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show(block=True)

    return 
