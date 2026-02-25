import os
from tqdm import tqdm

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from photutils.centroids import centroid_com


#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def get_images(data_dir, traces_included = ("+1",),
               verbose = 0):
    """Function to retrieve images and exposure times from data files.

    Args:
        data_dir (str): directory where the images you want to load are.
        traces_included (tup of str, optional): which traces' fluxes are
        included in the quicklook gif. Defaults to ("+1",).
        verbose (int, optional): how detailed you want the printed statements to be. Defaults to 0.

    Returns:
        np.array,np.array,np.array,np.array: images, exposure times, flux of the whole images, and flux from the section.
    """

    # initialize image and exposure time arrays
    images, exp_times, total_flux, partial_flux = [], [], [], []

    # get spectra directory
    specs_dir = os.path.join(data_dir, 'specimages/')

    # iterate over all files in directory
    have_xy = False
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

            # locate section
            if not have_xy:
                sections = []
                x, y = centroid_com(images[0]) # exploit the saturation of the 0th order to get in the vicinity of it
                x, y = (int(x), int(y))
                if "+1" in traces_included:
                    section = [y-20, y+40, x-350, x-100] # +1 order will always fall within these values, more or less
                    sections.append(section)
                if "-1" in traces_included:
                    section = [y-40, y+20, x+190, x+440] # -1 order will always fall within these values, more or less
                    sections.append(section)
                have_xy = True

            total_flux.append(np.sum(image))
            partial_flux_i = 0
            for section in sections:
                partial_flux_i += np.sum(image[section[0]:section[1],section[2]:section[3]])
            partial_flux.append(partial_flux_i)

    # convert to numpy arrays
    images = np.array(images)
    exp_times = np.array(exp_times)
    total_flux = np.array(total_flux)
    partial_flux = np.array(partial_flux)

    return images, exp_times, total_flux, partial_flux, sections


def parse_xarr(obs, traces_included = ("+1",),
               verbose = 0):
    """Function to retrieve images and exposure times from reduced xarray.

    Args:
        obs (xarray): xarray containing the reduced images.
        traces_included (tup of str, optional): which traces' fluxes are
        included in the quicklook gif. Defaults to ("+1",).
        verbose (int, optional): how detailed you want the printed statements to be. Defaults to 0.

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
    have_xy = False
    for i in tqdm(range(images.shape[0]),
                  desc='Parsing xarray for quicklookup... Progress:',
                  disable=(verbose < 2)):
        # locate section
            if not have_xy:
                sections = []
                x, y = centroid_com(images[0]) # exploit the saturation of the 0th order to get in the vicinity of it
                x, y = (int(x), int(y))
                if "+1" in traces_included:
                    section = [y-20, y+40, x-350, x-100] # +1 order will always fall within these values, more or less
                    sections.append(section)
                if "-1" in traces_included:
                    section = [y-40, y+20, x+190, x+440] # -1 order will always fall within these values, more or less
                    sections.append(section)
                have_xy = True

            total_flux.append(np.sum(images[i]))
            partial_flux_i = 0
            for section in sections:
                partial_flux_i += np.sum(images[i][section[0]:section[1],section[2]:section[3]])
            partial_flux.append(partial_flux_i)

    # convert to numpy arrays
    total_flux = np.array(total_flux)
    partial_flux = np.array(partial_flux)

    return images, dq, exp_times, total_flux, partial_flux, sections


def create_gif(exp_times, images, total_flux, partial_flux, sections,
               output_dir, show_fig = False, save_fig = False):
    """Function to create an animation showing all the exposures.

    Args:
        exp_times (np.array): exposure times for each image.
        images (np.array): array of 2D images to pull a light curve from.
        total_flux (np.array): flux summed across each 2D image.
        partial_flux (np.array): flux summed from the section of each 2D image.
        sections (lst of lst of int): the subsection(s) of image you measured flux in.
        output_dir (str): where to save the gif to.
        show_fig (bool, optional): whether to show the figure or not. Defaults to False.
        save_fig (bool, optional): wether to save the figure or not. Defaults to False.
    """

    # create copy to protect data from modification
    images = np.copy(images)

    # avoid zero and negative values for log plot
    images = org_images.copy()
    images[images <= 0] = 1e-7

    # create animation
    fig = plt.figure(figsize = (10, 7))
    gs = fig.add_gridspec(2, 2)
  
    # initialize exposure subplot and add exposure
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(images[0], norm='log', cmap = 'gist_gray', origin = 'lower', vmin = 0.1, vmax = 3000)
    # draw the box that defines the +1/-1 rough apertures
    for section in sections:
        rect = patches.Rectangle((section[2], section[0]), section[3] - section[2], section[1] - section[0], linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_xlabel('Detector x pixel')
    ax1.set_ylabel('Detector y pixel')

    # initialize Flux sum subplot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Total Image Flux', size = 10)
    sum_flux_line, = ax2.plot(exp_times, total_flux, '.', color = 'indianred')
    ax2.set_xlabel('Time of Exposure (BJD TDB)')
    ax2.set_ylabel('Counts (e-)')

    # initialize transit subplot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Boxed Region(s) Flux', size = 10)
    transit_line, = ax3.plot(exp_times, partial_flux, '.', color = 'indianred')
    ax3.set_xlabel('Time Of Exposure (MJD)')
    ax3.set_ylabel('Counts (counts)')
    

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
        sum_flux_line.set_data(exp_times[:i+1], total_flux[:i+1])

        # update line data 2
        transit_line.set_data(exp_times[:i+1], partial_flux[:i+1])
    
        return sum_flux_line, transit_line
        
    # create and plot animation
    animation = FuncAnimation(fig, animation_func, init_func = init,
                              frames = np.shape(images)[0], interval = 20)
    plt.tight_layout()
    if show_fig:
        plt.show(block = True)


    # save animation
    if save_fig:
        gif_dir = os.path.join(output_dir, 'quicklooks')

        if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)

        animation.save(os.path.join(gif_dir, 'quicklookup.gif'), writer = 'ffmpeg', fps = 10, dpi = 300)

    plt.close() # save memory

    return 


def create_dq_gif(exp_times, images, dq, sections,
                  output_dir, show_fig = False, save_fig = False):
    """Function to create an animation showing all the data quality flags.

    Args:
        exp_times (np.array): exposure times for each image.
        images (np.array): array of 2D images to pull a light curve from.
        dq (np.array): Array of 2D images showing pixels flagged for DQ.
        sections (lst of lst of int): the subsection(s) of image you measured flux in.
        output_dir (str): where to save the gif to.
        show_fig (bool, optional): whether to show the figure or not. Defaults to False.
        save_fig (bool, optional): wether to save the figure or not. Defaults to False.
    """

    # create animation
    fig = plt.figure(figsize = (10, 7))
    gs = fig.add_gridspec(2, 2)
  
    # initialize exposure subplot and add exposure
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(dq[0], norm='linear', cmap = 'gist_gray', origin = 'lower', vmin = 0, vmax = 1)
    # draw the box that defines the +1/-1 rough apertures
    for section in sections:
        rect = patches.Rectangle((section[2], section[0]), section[3] - section[2], section[1] - section[0], linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    ax1.add_patch(rect)
    ax1.set_xlabel('Detector X-Pixel')
    ax1.set_ylabel('Detector Y-Pixel')

    # initialize total Dq flags per frame subplot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Flags Per Frame', size = 10)
    dq_flags_per_frame = np.empty_like(exp_times)
    for k in range(dq.shape[0]):
        dq_flags_per_frame[k] = np.count_nonzero(dq[k,:,:])
    sum_flux_line, = ax2.plot(exp_times, dq_flags_per_frame, '.', color = 'indianred')
    ax2.set_xlabel('Time of Exposure (BJD TDB)')
    ax2.set_ylabel('Flags (N)')

    # initialize transit subplot
    ax3 = fig.add_subplot(gs[1, 1])
    n_tot = (section[1]-section[0])*(section[3]-section[2])
    ax3.set_title('Flags in Boxed Region(s) (total in = {})'.format(n_tot), size = 10)
    dq_flags_per_box = np.empty_like(exp_times)
    for k in range(dq.shape[0]):
        dq_flags_per_box[k] = np.count_nonzero(dq[k,section[0]:section[1],section[2]:section[3]])
    transit_line, = ax3.plot(exp_times, dq_flags_per_box, '.', color = 'indianred')
    ax3.set_xlabel('Time Of Exposure (MJD)')
    ax3.set_ylabel('Flags (#)')
    

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
        sum_flux_line.set_data(exp_times[:i+1], dq_flags_per_frame[:i+1])

        # update line data 2
        transit_line.set_data(exp_times[:i+1], dq_flags_per_box[:i+1])
    
        return sum_flux_line, transit_line
        
    # create and plot animation
    animation = FuncAnimation(fig, animation_func, init_func = init,
                              frames = np.shape(images)[0], interval = 20)
    plt.tight_layout()
    if show_fig:
        plt.show(block = True)


    # save animation
    if save_fig:
        gif_dir = os.path.join(output_dir, 'quicklooks')

        if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)

        animation.save(os.path.join(gif_dir, 'quicklookupDQ.gif'), writer = 'ffmpeg', fps = 10, dpi = 300)

    plt.close() # save memory

    return 


def quicklookup(data_dir, traces_included = ("+1",),
                verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Wrapper for quicklookup functions.

    Args:
        data_dir (str or xarray): directory where the images you want to load are, or axarray containing the data already reduced.
        traces_included (tup of str, optional): which traces' fluxes are included in the quicklook gif. Defaults to ("+1",).
        verbose (int, optional): how detailed you want the printed statements to be. Defaults to 0.
        show_plots (int, optional): how many plots you want to display. Defaults to 0.
        save_plots (int, optional): how many plots you want to save. Defaults to 0.
        output_dir (str, optional): directory where the gif should be saved, if save_plots >= 1. Defaults to None.
    """

    # get images and exposure times
    if isinstance(data_dir, str):
        images, exp_times, total_flux, partial_flux, sections = get_images(data_dir,
                                                                           traces_included,
                                                                           verbose)
        have_dq = False
    else:
        images, dq, exp_times, total_flux, partial_flux, sections = parse_xarr(data_dir,
                                                                               traces_included,
                                                                               verbose)
        have_dq = True
    
    # create animation gif
    save_fig = False
    if save_plots >= 1:
        save_fig = True
    show_fig = False
    if show_plots >= 1:
        show_fig = True
    create_gif(exp_times, images, total_flux, partial_flux, sections, output_dir,
               show_fig=show_fig, save_fig=save_fig)
    if have_dq:
        create_dq_gif(exp_times, images, dq, sections, output_dir,
                      show_fig=show_fig, save_fig=save_fig)
    
    return 
