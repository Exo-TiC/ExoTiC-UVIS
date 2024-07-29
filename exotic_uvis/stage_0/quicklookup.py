import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.pyplot import rc
from scipy import optimize



def get_images(data_dir, section):

    """
    
    Function to retrieve images and exposure times from data files
    
    """

    # initialize image and exposure time arrays
    images, exp_times, total_flux, partial_flux = [], [], [], []

    # get spectra directory
    specs_dir = os.path.join(data_dir, 'specimages/')

    # iterate over all files in directory
    for filename in np.sort(os.listdir(specs_dir)):

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

    return images, exp_times, total_flux, partial_flux




def get_transit(exp_times, images):

    """
    
    Function to get a raw transit 
    
    """
    return 0




def create_gif(exp_times, images, total_flux, partial_flux, section, output_dir, save_fig = False):


    """
    
    Function to create an animation showing all the exposures
    
    """


    # avoid zero and negative values for log plot
    images[images <= 0] = 1e-7

    # create animation
    fig = plt.figure(figsize = (10, 7))
    gs = fig.add_gridspec(2, 2)
    #fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=None, hspace = None)

  
    # initialize exposure subplot
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(np.log10(images[0]), cmap = 'gist_gray', origin = 'lower', vmin = -1, vmax = 3.5)
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
        sum_flux_line.set_data(exp_times[0], total_flux[0])
        transit_line.set_data(exp_times[0], partial_flux[0])

        return sum_flux_line, transit_line

    # define animation function
    def animation_func(i):

        # update image data
        im.set_data(np.log10(images[i]))

        # update line data
        sum_flux_line.set_data(exp_times[:i], total_flux[:i])

        # update line data 2
        transit_line.set_data(exp_times[:i], partial_flux[:i])
    
        return sum_flux_line, transit_line, 
        
    # create and plot animation
    animation = FuncAnimation(fig, animation_func, init_func = init, frames = np.shape(images)[0], interval = 20)
    plt.tight_layout()
    plt.show(block = True)


    # save animation
    if save_fig:
        stage0dir = os.path.join(output_dir, 'stage0/')

        if not os.path.exists(stage0dir):
                os.makedirs(stage0dir)

        animation.save(os.path.join(stage0dir, 'quicklookup.gif'), writer = 'ffmpeg', fps = 10)

    plt.close() # save memory

    return 0



def quicklookup(data_dir, output_dir):

    # define partial section
    section = [280, 350, 700, 950]

    # get images and exposure times
    images, exp_times, total_flux, partial_flux = get_images(data_dir, section)

    # get transit
    #get_transit(exp_times, images)

    # create animation gif
    create_gif(exp_times, images, total_flux, partial_flux, section, output_dir, save_fig=False)


    return 0




