import os

import matplotlib.pyplot as plt


#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def plot_exposure(images, line_data = None, scatter_data = None, 
                  extent = None, xlimits = None, ylimits = None, title = None, 
                  min = 1e-3, max = 1e4, mark_size = 30, linestyles = None,
                  cbar_orient='vertical', cbar_aspect=20, cbar_pad = 0.05, cbar_fraction=0.15,
                  show_plot = False, save_plot = False, 
                  filename = None, output_dir = None):
    """Function to plot an image given certain parameters.

    Args:
        images (np.array): images from the obs.images.
        line_data (list, optional): list of lists of x, y values denoting lines you want to draw on the plot. Defaults to None.
        scatter_data (list, optional): list of x, y points you want to scatter on the plot. Defaults to None.
        extent (tuple of float, optional): if not None, defines bounds of array you want to plot. Defaults to None.
        xlimits (tuple of float, optional): if not None, defines x axis limits of array you want to plot. Defaults to None.
        ylimits (tuple of float, optional): if not None, defines y axis limits of array you want to plot. Defaults to None.
        title (str, optional): title for the plot. Defaults to None.
        min (int, optional): darkest point for the colormap. Defaults to 1e-3.
        max (int, optional): brightest point for the colormap. Defaults to 1e4.
        mark_size (int, optional): size of scatter points, if scatter_data is not None. Defaults to 30.
        linestyles (list of str, optional): style for each line, if line_data is not None. Defaults to None.
        cbar_orient (str, optional): colorbar orientation. Defaults to 'vertical'.
        cbar_aspect (float, optional): colorbar aspect ratio. Defaults to 20.
        cbar_fraction (float, optional): colorbar axis fraction. Defaults to 0.15.
        cbar_pad (float, optional): colorbar padding. Defaults to 0.05.
        show_plot (bool, optional): whether to interrupt execution to show the user the plot. Defaults to False.
        save_plot (bool, optional): whether to save this plot. Defaults to False.
        filename (list of str, optional): name for each plot file. Defaults to None.
        output_dir (str, optional): where to save the plots to, if save_plots is True. Defaults to None.
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
        plt.colorbar(orientation=cbar_orient,aspect=cbar_aspect,
                     pad=cbar_pad,fraction=cbar_fraction)

        if line_data:
            if linestyles is None:
                linestyles = ['-' for line in line_data]
            for line, ls in zip(line_data,linestyles):
                plt.plot(line[0], line[1], color='indianred', ls=ls)

        if scatter_data: 
            plt.scatter(scatter_data[0], scatter_data[1], s = mark_size, color = 'r', marker = '+')
        
        if xlimits:
            plt.xlim(xlimits[0],xlimits[1])
        if ylimits:
            plt.ylim(ylimits[0],ylimits[1])

        if title:
            plt.title(title)
        
        if save_plot:
            plot_dir = os.path.join(output_dir, 'plots')
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir) 
            filedir = os.path.join(plot_dir, f'{filename[i]}.png')
            plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)
        
        if show_plot:
            plt.show(block=True)

        plt.close() # save memory
    
    return
