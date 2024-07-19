import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import rc
from scipy import optimize



def get_images(data_dir):

    """
    
    Function to retrieve images and exposure times from data files
    
    """

    # initialize image and exposure time arrays
    images, exp_times, sum_flux = [], [], []

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
            sum_flux.append(np.sum(image))

    # convert to numpy arrays
    images = np.array(images)
    exp_times = np.array(exp_times)
    sum_flux = np.array(sum_flux)

    return images, exp_times




def get_transit(exp_times, images):

    """
    
    Function to get a raw transit 
    
    """







    return 0




def create_gif(exp_times, images, output_dir):


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
    ax1.set_xlabel('Detector x pixel')
    ax1.set_ylabel('Detector y pixel')

    # initialize Flux sum subplot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Image Total Flux', size = 10)
    sum_flux = ax2.scatter([], [], color = 'indianred')
    ax2.set_xlabel('Time of Exposure (BJD TDB)')
    ax2.set_ylabel('Counts (e-)')

    # initialize transit subplot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Raw White-light Curve (+1 Order)', size = 10)
    raw_lc = ax3.scatter([], [], color = 'indianred')
    ax3.set_xlabel('Time of Exposure (BJD TDB)')
    ax3.set_ylabel('Counts (e-)')
    

    # initialize 
    def init():

        return 

    # define animation function
    def animation_func(i):

        # update image data
        im.set_data(np.log10(images[i]))
    
        return 
        
    # create and plot animation
    animation = FuncAnimation(fig, animation_func, init_func = init, frames = np.shape(images)[0], interval = 20)
    plt.tight_layout()
    plt.show()

    # save animation
    animation.save(os.path.join(output_dir, 'quicklookup.gif'), writer = 'ffmpeg', fps = 10)

    return 0



def quicklookup(data_dir, output_dir):

    # get images and exposure times
    images, exp_times = get_images(data_dir)

    # get transit
    #get_transit(exp_times, images)

    # create animation gif
    create_gif(exp_times, images, output_dir)


    return 0




