import os
from tqdm import tqdm
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.pyplot import rc
from scipy import optimize
from photutils.centroids import centroid_com



def get_images(data_dir, section,
               verbose = 0):
    """Function to retrieve images and exposure times from data files.

    Args:
        data_dir (str): Directory where the images you want to load are.
        section (lst of int): The subsection of image you want to measure flux in.
        verbose (int, optional): How detailed you want the printed statements to be. Defaults to 0.

    Returns:
        np.array,np.array,np.array,np.array: images, exposure times, flux of the whole images, and flux from the section.
    """

    # initialize image and exposure time arrays
    images, exp_times, total_flux, partial_flux = [], [], [], []

    # get spectra directory
    specs_dir = os.path.join(data_dir, 'specimages/')

    # iterate over all files in directory
    for filename in tqdm(np.sort(os.listdir(specs_dir)),
                         desc='Parsing files for quicklookup... Progress:',
                         disable=(verbose < 2)):

        # open flt files but avoid f_flt files (embedded) if created
        if (filename[-9:] == '_flt.fits') and (filename[-10] != 'f'):

            # open fits and image data
            hdul = fits.open(os.path.join(specs_dir, filename))
            image = np.array(hdul[1].data)

            # append image and exposure data
            exp_times.append((hdul[0].header['EXPSTART'] + hdul[0].header['EXPEND'])/2)
            images.append(image) 
            total_flux.append(np.sum(image))
            partial_flux.append(np.sum(image[section[0]:section[1],section[2]:section[3]]))

    # convert to numpy arrays
    images = np.array(images)
    exp_times = np.array(exp_times)
    total_flux = np.array(total_flux)
    partial_flux = np.array(partial_flux)

    # locate section
    x, y = centroid_com(images[0]) # exploit the saturation of the 0th order to get in the vicinity of it
    x, y = (int(x), int(y))
    section = [y-20, y+40, x-350, x-100] # +1 order will always fall within these values, more or less

    return images, exp_times, total_flux, partial_flux, section

def parse_xarr(obs, section,
               verbose = 0):
    """Function to retrieve images and exposure times from reduced xarray.

    Args:
        obs (xarray): xarray containing the reduced images.
        section (lst of int): The subsection of image you want to measure flux in.
        verbose (int, optional): How detailed you want the printed statements to be. Defaults to 0.

    Returns:
        np.array,np.array,np.array,np.array: images, exposure times, flux of the whole images, and flux from the section.
    """

    # initialize flux arrays
    total_flux, partial_flux = [], []

    # get images and exp_times directly
    images = obs.images.data
    exp_times = obs.exp_time.data

    # get data quality
    dq = obs.data_quality.data

    # iterate over all files in directory
    for i in tqdm(range(images.shape[0]),
                  desc='Parsing xarray for quicklookup... Progress:',
                  disable=(verbose < 2)):

        # append total and partial fluxes
        total_flux.append(np.sum(images[i]))
        partial_flux.append(np.sum(images[i][section[0]:section[1],section[2]:section[3]]))

    # convert to numpy arrays
    total_flux = np.array(total_flux)
    partial_flux = np.array(partial_flux)

    # locate section
    x, y = centroid_com(images[0]) # exploit the saturation of the 0th order to get in the vicinity of it
    x, y = (int(x), int(y))
    section = [y-20, y+40, x-350, x-100] # +1 order will always fall within these values, more or less

    return images, dq, exp_times, total_flux, partial_flux, section


def get_transit(exp_times, images):
    """Function to get a raw transit.

    Args:
        exp_times (np.array): Exposure times for each image.
        images (np.array): Array of 2D images to pull a light curve from.

    Returns:
        int: Placeholder.
    """
    return 0




def create_gif(exp_times, images, total_flux, partial_flux, section,
               output_dir, stage, show_fig = False, save_fig = False):
    """Function to create an animation showing all the exposures.

    Args:
        exp_times (np.array): Exposure times for each image.
        images (np.array): Array of 2D images to pull a light curve from.
        total_flux (np.array): _description_
        partial_flux (np.array): _description_
        section (lst of int): The subsection of image you want to measure flux in.
        output_dir (str): Where to save the gif to.
        stage (str): Which stage this quicklook is for.
        show_fig (bool, optional): Whether to show the figure or not. Defaults to False.
        save_fig (bool, optional): Whether to save the figure or not. Defaults to False.
    """
    # avoid zero and negative values for log plot
    images[images <= 0] = 1e-7

    # create animation
    fig = plt.figure(figsize = (10, 7))
    gs = fig.add_gridspec(2, 2)
    #fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=None, hspace = None)
  
    # initialize exposure subplot and add exposure
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(images[0], norm='log', cmap = 'gist_gray', origin = 'lower', vmin = 0.1, vmax = 3000)
    # draw the box that defines the +1 rough aperture
    rect = patches.Rectangle((section[2], section[0]), section[3] - section[2], section[1] - section[0], linewidth=1, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_xlabel('Detector x pixel')
    ax1.set_ylabel('Detector y pixel')

    # initialize Flux sum subplot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Total Image Flux', size = 10)
    sum_flux_line, = ax2.plot(exp_times, total_flux, '.', color = 'indianred')
    #sum_flux_line,  = ax2.plot([], [], '.', color = 'indianred')
    ax2.set_xlabel('Time of Exposure (BJD TDB)')
    ax2.set_ylabel('Counts (e-)')

    # initialize transit subplot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Box image Flux', size = 10)
    transit_line, = ax3.plot(exp_times, partial_flux, '.', color = 'indianred')
    ax3.set_xlabel('Time of Exposure (BJD TDB)')
    ax3.set_ylabel('Counts (e-)')
    

    # initialize 
    def init():
        sum_flux_line.set_data([exp_times[0],], [total_flux[0],])
        transit_line.set_data([exp_times[0],], [partial_flux[0],])

        return sum_flux_line, transit_line

    # define animation function
    def animation_func(i):
        # update image data
        im.set_data(images[i])

        # update line data
        sum_flux_line.set_data(exp_times[:i], total_flux[:i])

        # update line data 2
        transit_line.set_data(exp_times[:i], partial_flux[:i])
    
        return sum_flux_line, transit_line
        
    # create and plot animation
    animation = FuncAnimation(fig, animation_func, init_func = init, frames = np.shape(images)[0], interval = 20)
    plt.tight_layout()
    if show_fig:
        plt.show(block = True)


    # save animation
    if save_fig:
        stage0dir = os.path.join(output_dir, '{}/'.format(stage))

        if not os.path.exists(stage0dir):
                os.makedirs(stage0dir)

        animation.save(os.path.join(stage0dir, 'quicklookup.gif'), writer = 'ffmpeg', fps = 10)

    plt.close() # save memory


def create_dq_gif(exp_times, images, dq, section,
                  output_dir, stage, show_fig = False, save_fig = False):
    """Function to create an animation showing all the data quality flags.

    Args:
        exp_times (np.array): Exposure times for each image.
        images (np.array): Array of 2D images to pull a light curve from.
        dq (np.array): Array of 2D images showing pixels flagged for DQ.
        section (lst of int): The subsection of image you want to measure flux in.
        output_dir (str): Where to save the gif to.
        stage (str): Which stage this quicklook is for.
        show_fig (bool, optional): Whether to show the figure or not. Defaults to False.
        save_fig (bool, optional): Whether to save the figure or not. Defaults to False.
    """

    # create animation
    fig = plt.figure(figsize = (10, 7))
    gs = fig.add_gridspec(2, 2)
    #fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=None, hspace = None)
  
    # initialize exposure subplot and add exposure
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(dq[0], norm='linear', cmap = 'gist_gray', origin = 'lower', vmin = 0, vmax = 1)
    # draw the box that defines the +1 rough aperture
    rect = patches.Rectangle((section[2], section[0]), section[3] - section[2], section[1] - section[0], linewidth=1, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_xlabel('Detector x pixel')
    ax1.set_ylabel('Detector y pixel')

    # initialize total Dq flags per frame subplot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Flags per frame', size = 10)
    dq_flags_per_frame = np.empty_like(exp_times)
    for k in range(dq.shape[0]):
        dq_flags_per_frame[k] = np.count_nonzero(dq[k,:,:])
    sum_flux_line, = ax2.plot(exp_times, dq_flags_per_frame, '.', color = 'indianred')
    #sum_flux_line,  = ax2.plot([], [], '.', color = 'indianred')
    ax2.set_xlabel('Time of Exposure (BJD TDB)')
    ax2.set_ylabel('Flags (N)')

    # initialize transit subplot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Flags per Box', size = 10)
    dq_flags_per_box = np.empty_like(exp_times)
    for k in range(dq.shape[0]):
        dq_flags_per_box[k] = np.count_nonzero(dq[k,section[0]:section[1],section[2]:section[3]])
    transit_line, = ax3.plot(exp_times, dq_flags_per_box, '.', color = 'indianred')
    ax3.set_xlabel('Time of Exposure (BJD TDB)')
    ax3.set_ylabel('Flags (N)')
    

    # initialize 
    def init():
        sum_flux_line.set_data([exp_times[0],], [dq_flags_per_frame[0],])
        transit_line.set_data([exp_times[0],], [dq_flags_per_box[0],])

        return sum_flux_line, transit_line

    # define animation function
    def animation_func(i):
        # update image data
        im.set_data(dq[i])

        # update line data
        sum_flux_line.set_data(exp_times[:i], dq_flags_per_frame[:i])

        # update line data 2
        transit_line.set_data(exp_times[:i], dq_flags_per_box[:i])
    
        return sum_flux_line, transit_line
        
    # create and plot animation
    animation = FuncAnimation(fig, animation_func, init_func = init, frames = np.shape(images)[0], interval = 20)
    plt.tight_layout()
    if show_fig:
        plt.show(block = True)


    # save animation
    if save_fig:
        stage0dir = os.path.join(output_dir, '{}/'.format(stage))

        if not os.path.exists(stage0dir):
                os.makedirs(stage0dir)

        animation.save(os.path.join(stage0dir, 'quicklookupDQ.gif'), writer = 'ffmpeg', fps = 10)

    plt.close() # save memory

def quicklookup(data_dir,
                verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Wrapper for quicklookup functions.

    Args:
        data_dir (str or xarray): Directory where the images you want to load are,
        or axarray containing the data already reduced.
        verbose (int, optional): How detailed you want the printed statements to be. Defaults to 0.
        show_plots (int, optional): How many plots you want to display. Defaults to 0.
        save_plots (int, optional): How many plots you want to save. Defaults to 0.
        output_dir (str, optional): Directory where the gif should be saved, if save_plots >= 1. Defaults to None.
    """

    # define partial section
    section = [280, 350, 700, 950] # FIX: possibly redundant now?

    # get images and exposure times
    if isinstance(data_dir, str):
        stage = "stage0"
        images, exp_times, total_flux, partial_flux, section = get_images(data_dir, section, verbose)
        have_dq = False
    else:
        stage = "stage1"
        images, dq, exp_times, total_flux, partial_flux, section = parse_xarr(data_dir, section, verbose)
        have_dq = True

    # get transit
    #get_transit(exp_times, images)

    # create animation gif
    save_fig = False
    if save_plots >= 1:
        save_fig = True
    show_fig = False
    if show_plots >= 1:
        show_fig = True
    create_gif(exp_times, images, total_flux, partial_flux, section, output_dir, stage=stage, show_fig=show_fig, save_fig=save_fig)
    if have_dq:
        create_dq_gif(exp_times, images, dq, section, output_dir, stage=stage, show_fig=show_fig, save_fig=save_fig)