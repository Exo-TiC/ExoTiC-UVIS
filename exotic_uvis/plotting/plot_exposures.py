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
                  show_plot = False, save_plot = False, 
                  stage = 0, filename = None, output_dir = None):
    """Function to plot an image given certain parameters 

    Args:
        images (_type_): _description_
        line_data (_type_, optional): _description_. Defaults to None.
        scatter_data (_type_, optional): _description_. Defaults to None.
        extent (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.
        min (int, optional): _description_. Defaults to -3.
        max (int, optional): _description_. Defaults to 4.
        mark_size (int, optional): _description_. Defaults to 30.
        show_plot (bool, optional): _description_. Defaults to False.
        save_plot (bool, optional): _description_. Defaults to False.
        stage (int, optional): _description_. Defaults to 0.
        filename (_type_, optional): _description_. Defaults to None.
        output_dir (_type_, optional): _description_. Defaults to None.
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

        stagedir = os.path.join(output_dir, f'stage{stage}/plots/')

        if not os.path.exists(stagedir):
                os.makedirs(stagedir) 
        
        filedir = os.path.join(stagedir, f'{filename[i]}.png')
        
        if save_plot:
            plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)
        
    if show_plot:
        plt.show(block=True)
    
    return
