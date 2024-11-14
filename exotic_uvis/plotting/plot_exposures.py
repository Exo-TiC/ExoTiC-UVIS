import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr


#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def plot_exposure(images, line_data = None, scatter_data = None, 
                  extent = None, title = None, 
                  min = 1e-3, max = 1e4, mark_size = 30,
                  show_plot = False, save_plot = False, 
                  stage = 0, filename = None, output_dir = None):
    """Function to plot an image given certain parameters .

    Args:
        images (np.array): images from the obs.images.
        line_data (list, optional): list of lists of x, y values denoting
        lines you want to draw on the plot. Defaults to None.
        scatter_data (list, optional): list of x, y points you want to
        scatter on the plot. Defaults to None.
        extent (tuple of float, optional): if not None, defines bounds
        of array you want to plot. Defaults to None.
        title (str, optional): title for the plot. Defaults to None.
        min (int, optional): darkest point for the colormap. Defaults to 1e-3.
        max (int, optional): brightest point for the colormap. Defaults to 1e4.
        mark_size (int, optional): size of scatter points, if scatter_data
        is not None. Defaults to 30.
        show_plot (bool, optional): whether to interrupt execution to show
        the user the plot. Defaults to False.
        save_plot (bool, optional): whether to save this plot. Defaults to False.
        stage (int, optional): which stage this is being executed in, used for
        giving files proper names. Defaults to 0.
        filename (list of str, optional): name for each plot file. Defaults to None.
        output_dir (str, optional): where to save the plots to, if save_plots is
        True. Defaults to None.
    """
    
    for i, data in enumerate(images): 

        image = data.copy()
        image[image <= 0] = 1e-10 # allows us to use log normalization

        plt.figure(figsize = (20, 4))
        plt.imshow(image, origin = 'lower', norm='log', 
                   vmin = min, vmax = max, 
                   cmap = 'gist_gray', extent = extent)
        plt.xlabel('Detector x-pixel')
        plt.ylabel('Detector y-pixel')
        plt.colorbar()

        if line_data:
            for line in line_data:
                plt.plot(line[0], line[1])

        if scatter_data: 
            plt.scatter(scatter_data[0], scatter_data[1], s = mark_size, color = 'r', marker = '+')

        if title:
            plt.title(title)
        
        if save_plot:
            stagedir = os.path.join(output_dir, f'stage{stage}/plots/')
            if not os.path.exists(stagedir):
                os.makedirs(stagedir) 
            filedir = os.path.join(stagedir, f'{filename[i]}.png')
            plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)
        
        if show_plot:
            plt.show(block=True)

        plt.close() # save memory
    
    return
