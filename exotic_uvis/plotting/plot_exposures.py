import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr
import os

#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def plot_exposure(images, line_data = None, scatter_data = None, 
                  extent = None, title = None, 
                  min = -3, max = 4, mark_size = 30, 
                  show = True, stage = 0, filename = None, output_dir = None):

    """
    Function to plot an image given certain parameters 
    
    """

    for i, data in enumerate(images): 

        image = data.copy()
        image[image <= 0] = 1e-10

        plt.figure(figsize = (20, 4))
        plt.imshow(np.log10(image), origin = 'lower', vmin = min, vmax = max, cmap = 'gist_gray', extent = extent)
        plt.xlabel('Detector x-pixel')
        plt.ylabel('Detector y-pixel')
        plt.colorbar()

        if line_data:
            for j, line in enumerate(line_data):
                plt.plot(line[0], line[1])

        if scatter_data: 
            plt.scatter(scatter_data[0], scatter_data[1], s = mark_size, color = 'r', marker = '+')

        if title:
            plt.title(title)

        #if filename:
        stagedir = os.path.join(output_dir, f'stage{stage}/plots/')

        if not os.path.exists(stagedir):
                os.makedirs(stagedir) 
        
        filedir = os.path.join(stagedir, f'{filename[i]}.png')
        plt.savefig(filedir, bbox_inches = 'tight')
        
    if show:
        plt.show()
    
    return


def plot_corners(images, corners, output_dir = None):

    """
    
    Function to plot exposure with rectangles to indicate the corners used for background subtraction
    
    """

    plot_exposure(images, show = False)
    ax = plt.gca()

    for corner in corners:
        rect = patches.Rectangle((corner[2], corner[0]), corner[3] - corner[2], 
                                 corner[1] - corner[0], linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)
    plt.show()

    return 
