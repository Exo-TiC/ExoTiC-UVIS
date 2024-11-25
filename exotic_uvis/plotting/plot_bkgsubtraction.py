import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def plot_corners(image, corners, 
                 min = 1e-3, max = 1e4,
                 show_plot = False, save_plot = False, 
                 output_dir = None):
    """Function to plot exposure with rectangles to indicate the corners used
    for background subtraction.

    Args:
        image (np.array): 2D image from the obs xarray.
        corners (lst of lsts): x, y bounds of each rectangle used to define
        the corners from which the background is measured.
        min (int, optional): darkest point for the colormap. Defaults to 1e-3.
        max (int, optional): brightest point for the colormap. Defaults to 1e4.
        show_plot (bool, optional): whether to interrupt execution to show
        the user the plot. Defaults to False.
        save_plot (bool, optional): whether to save this plot. Defaults to False.
        output_dir (str, optional): output directory where the plot will be
        saved. Defaults to None.
    """
    
    image = image.copy()
    image[image <= 0] = 1e-10

    plt.figure(figsize = (20, 4))
    plt.imshow(image, origin = 'lower', norm='log', 
                vmin = min, vmax = max, 
                cmap = 'gist_gray')

    ax = plt.gca()

    # draw each corner region onto the exposure
    for corner in corners:
        rect = patches.Rectangle((corner[2], corner[0]), corner[3] - corner[2], 
                                 corner[1] - corner[0], linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)
    plt.xlabel('Detector x-pixel')
    plt.ylabel('Detector y-pixel')
    plt.colorbar()

    if save_plot:
        stagedir = os.path.join(output_dir, f'stage1/plots/')
        if not os.path.exists(stagedir):
            os.makedirs(stagedir) 
        filedir = os.path.join(stagedir, f'bkg_corner_location.png')
        plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)
        
    if show_plot:
        plt.show(block=True)
    else:
        plt.close() # save memory
    
    return 


def plot_bkgvals(exp_times, bkg_vals, method,
                 output_dir = None, save_plot = False, show_plot = False):
    """Function to create measured background value plots for all methods.

    Args:
        exp_times (np.array): BJD exposure times for each frame.
        bkg_vals (np.array): 1D or 2D array of measured background values.
        method (str): The method used for background subtraction, useful to
        distinguish each plot file from each other.
        output_dir (str, optional): output directory where the plot will be
        saved. Defaults to None.
        save_plot (bool, optional): whether to save the plot to a file.
        Defaults to False.
        show_plot (bool, optional): whether to interrupt execution to
        show the user the plot. Defaults to False.
    """

    # initialize figure
    plt.figure(figsize = (10, 7))
    if method != 'col-by-col':
        # if it's not col-by-col, we take a single bkg value per frame, 1D
        plt.plot(exp_times, bkg_vals, '-o', color='indianred')
        plt.xlabel('Exposure')
        plt.ylabel('Background Counts')
        plt.title('Image background per exposure')
        if method == 'Pagul':
            plt.ylabel('Pagul et al. image scaling parameter')
            plt.title('Scaling parameter per exposure')
    
    else:
        # if it's col-by-col, we take a bkg value per column per frame, 2D
        v = np.nanmedian(bkg_vals)
        plt.imshow(bkg_vals,aspect=20, vmin=0.5*v,vmax=1.5*v)
        plt.colorbar(fraction=0.01)
        plt.xlabel('Column #')
        plt.ylabel('Exposure index')
        plt.title("Image background columns by exposure")

    stagedir = os.path.join(output_dir, 'stage1/plots/') 
    filedir = os.path.join(stagedir, 'bkg_values_{}.png'.format(method))
    
    if save_plot:
        if not os.path.exists(stagedir):
            os.makedirs(stagedir) 
        plt.savefig(filedir, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show(block=True)
    
    plt.close() # save memory

    return 


def plot_mode_v_params(exp_times, modes, params,
                       output_dir = None, save_plot = False, show_plot = False):
    """Function to create a diagnostic plot for Pagul et al. bkg subtraction.

    Args:
        exp_times (np.array): BJD exposure times for each frame.
        modes (np.array): measured mode of each frame, used for comparison.
        params (np.array): Pagul+ sky image scaling parameter. Ideally, the
        mode and scaling parameters should not be too different.
        output_dir (str, optional): output directory where the plot will be
        saved. Defaults to None.
        save_plot (bool, optional): whether to save the plot to a file.
        Defaults to False.
        show_plot (bool, optional): whether to interrupt execution to
        show the user the plot. Defaults to False.
    """

    # initialize figure
    plt.figure(figsize = (10, 7))
    # add the modes and params in different colors and markers
    plt.scatter(exp_times, modes, marker='s', color='red',label='mode')
    plt.scatter(exp_times, params, marker='o', color='k',label='scaling parameter')
    plt.xlabel('Exposure')
    plt.ylabel('Counts')
    plt.title('Frame mode vs scaling parameter')
    plt.legend()
    
    stagedir = os.path.join(output_dir, 'stage1/plots/') 
    filedir = os.path.join(stagedir, 'bkg_scaling_parameters.png')
    
    if save_plot:
        if not os.path.exists(stagedir):
            os.makedirs(stagedir) 
        plt.savefig(filedir, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show(block=True)
    
    plt.close() # save memory

    return 


def plot_histogram(bin_cents, array, mode, median, hist_min, hist_max, hist_bins, fit, exp_num, 
                   gaussian_center=False, gaussian_fit = None, show_plots=0, save_plots=0, output_dir=None):
    """_summary_

    Args:
        bin_cents (_type_): _description_
        array (_type_): _description_
        mode (_type_): _description_
        median (_type_): _description_
        hist_min (_type_): _description_
        hist_max (_type_): _description_
        hist_bins (_type_): _description_
        fit (_type_): _description_
        exp_num (_type_): _description_
        gaussian_center (bool, optional): _description_. Defaults to False.
        gaussian_fit (_type_, optional): _description_. Defaults to None.
        show_plots (int, optional): _description_. Defaults to 0.
        save_plots (int, optional): _description_. Defaults to 0.
        output_dir (_type_, optional): _description_. Defaults to None.
    """


    plt.figure(figsize = (10, 7))
    plt.hist(array, bins = bin_cents, color = 'indianred', alpha = 0.7, density=False)

    if gaussian_center:
        plt.axvline(gaussian_center, linestyle = '--', color = 'red', label='Gaussian center')
        plt.plot(bin_cents, gaussian_fit, color = 'gray')

    plt.axvline(median, linestyle = '--', color = 'dodgerblue', label='Mode')
    plt.axvline(mode, linestyle = '--', color = 'gold', label='Median')
    plt.xlabel('Pixel Value')
    plt.ylabel('Counts')
    plt.legend()
    plt.title(f'Background Values histogram Exposure {exp_num}')

    if save_plots:
        stagedir = os.path.join(output_dir, f'stage1/plots/')
        if not os.path.exists(stagedir):
            os.makedirs(stagedir) 
        filedir = os.path.join(stagedir, f'bkg_histogram_exposure{exp_num}.png')
        plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)
    
    if show_plots:
        plt.show(block=True)

    plt.close() # save memory



    return