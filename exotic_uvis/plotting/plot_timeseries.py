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
        series_x (_type_): _description_
        series_y (_type_): _description_
        style (_type_): _description_
        line_data (_type_, optional): _description_. Defaults to None.
        scatter_data (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.
        xlabel (_type_, optional): _description_. Defaults to None.
        ylabel (_type_, optional): _description_. Defaults to None.
        xmin (int, optional): _description_. Defaults to 0.
        xmax (_type_, optional): _description_. Defaults to 1.
        ymin (int, optional): _description_. Defaults to 0.
        ymax (_type_, optional): _description_. Defaults to 1e4.
        mark_size (int, optional): _description_. Defaults to 30.
        line_style (str, optional): _description_. Defaults to '-'.
        show_plot (bool, optional): _description_. Defaults to False.
        save_plot (bool, optional): _description_. Defaults to False.
        stage (int, optional): _description_. Defaults to 0.
        filename (_type_, optional): _description_. Defaults to None.
        output_dir (_type_, optional): _description_. Defaults to None.
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