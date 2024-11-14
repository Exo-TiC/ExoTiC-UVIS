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

def plot_flags_per_time(series_x, series_y, style='line',
                        line_data = None, scatter_data = None, 
                        title = None, xlabel=None, ylabel=None,
                        xmin = 0, xmax = 1, ymin = 0, ymax = 1e4,
                        mark_size = 30, line_style='-',
                        show_plot = False, save_plot = False, 
                        stage = 0, filename = None, output_dir = None):
    """Function to plot number of flagged pixels vs time.

    Args:
        series_x (array-like): series of x coordinates for plotting.
        series_y (array-like): series of y coordinates for plotting.
        style (str): options are 'line' or 'scatter'. Defaults to 'line'.
        line_data (array-like, optional): x, y values defining lines
        to overplot on top of series_x, series_y. Defaults to None.
        scatter_data (array-like, optional): x, y values defining scatter
        points to overplot on top of series_x, series_y. Defaults to None.
        title (str, optional): title for the plot. Defaults to None.
        xlabel (str, optional): x axis label. Defaults to None.
        ylabel (str, optional): y axis label. Defaults to None.
        xmin (float, optional): x axis lower limit. Defaults to 0.
        xmax (float, optional): x axis upper limit. Defaults to 1.
        ymin (float, optional): y axis lower limit. Defaults to 0.
        ymax (float, optional): y axis upper limit. Defaults to 1e4.
        mark_size (float, optional): size of scatter points. Defaults to 30.
        line_style (str, optional): mpl style of line. Defaults to '-'.
        show_plot (bool, optional): whether to interrupt execution to show the
        user the plot. Defaults to False.
        save_plot (bool, optional): whether to save this plot to a file.
        Defaults to False.
        stage (int, optional): which stage this is being executed in, for
        file naming purposes. Defaults to 0.
        filename (list of str, optional): names to give each output file,
        if saving. Defaults to None.
        output_dir (str, optional): where to save the files to. Defaults to None.
    """
    
    for i, (x, y) in enumerate(zip(series_x, series_y)):

        plt.figure(figsize = (20, 4))
        if style == 'scatter':
            plt.scatter(x, y, color='k', s=mark_size)
        elif style == 'line':
            plt.plot(x, y, color='k', ls=line_style)
        if xlabel:
            plt.xlabel(xlabel[i])
        if ylabel:
            plt.ylabel(ylabel[i])
        plt.colorbar()

        if xmin or xmax:
            plt.xlim(xmin, xmax)

        if ymin or ymax:
            plt.ylim(ymin, ymax)

        if line_data:
            for j, line in enumerate(line_data):
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


def plot_light_curve():
    # Placeholder
    return "yo"