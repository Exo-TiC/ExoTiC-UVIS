import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import xarray as xr

#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def plot_one_spectrum(wavelengths, spectrum, order="+1",
                      stage = 2, show_plot = False, save_plot = False,
                      filename = None, output_dir = None):
    """Function to plot one extracted spectrum

    Args:
        wavelengths (_type_): _description_
        spectrum (_type_): _description_
        order (str, optional): _description_. Defaults to "+1".
        show_plot (bool, optional): _description_. Defaults to False.
        save_plot (bool, optional): _description_. Defaults to False.
        filename (_type_, optional): _description_. Defaults to None.
        output_dir (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    ok = (wavelengths>2000) & (wavelengths<8000)

    plt.figure(figsize = (10, 7))
    plt.plot(wavelengths[ok], spectrum[ok])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Extracted Counts')
    plt.title('Example of extracted order {} spectrum'.format(order))
    
    if save_plot:
        stagedir = os.path.join(output_dir, f'stage{stage}/plots/')
        if not os.path.exists(stagedir):
            os.makedirs(stagedir) 
        filedir = os.path.join(stagedir, f'{filename}.png')
        plt.savefig(filedir,dpi=300,bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close() # save memory

    return 0

def plot_spec_gif(wavelengths, spectra, orders=("+1",),
                  stage = 2, show_plot = False, save_plot = False,
                  filename = None, output_dir = None):
    """Plots a gif of the extracted spectra over time.

    Args:
        wavelengths (_type_): _description_
        spectra (_type_): _description_
        order (str, optional): _description_. Defaults to "+1".
        show_plot (bool, optional): _description_. Defaults to False.
        save_plot (bool, optional): _description_. Defaults to False.
        filename (_type_, optional): _description_. Defaults to None.
        output_dir (_type_, optional): _description_. Defaults to None.
    """
    # define order colors
    colors = {"+1":'red',"-1":'blue',
              "+2":'orangered',"-2":'royalblue',
              "+3":'darkorange',"-3":'dodgerblue',
              "+4":'orange',"-4":'deepskyblue'}

    # create animation
    fig,ax = plt.subplots(nrows=len(spectra),figsize = (10, 7*len(spectra)))
    spec_lines = []
  
    # plot first spectrum on each axis get things started
    for n, order in enumerate(orders):
        ok = (wavelengths[n]>2000) & (wavelengths[n]<8000)
        spec_line = ax[n].plot(wavelengths[n][ok],spectra[n,0,ok],color = colors[order])
        spec_lines.append(spec_line)
        ax[n].set_xlim(2000,8000)
        ax[n].set_ylim(0, np.nanmax(spectra[n,:,ok]))
        ax[n].set_title("{} order, frame 0".format(order))
        ax[n].set_xlabel('wavelength [AA]')
        ax[n].set_ylabel('counts [a.u.]')

    # initialize 
    def init():
        for n, order in enumerate(orders):
            ok = (wavelengths[n]>2000) & (wavelengths[n]<8000)
            spec_lines[n][0].set_data([wavelengths[n][ok],spectra[n,0,ok]])
            ax[n].set_title("{} order, frame {}".format(order,0))

        return spec_lines

    # define animation function
    def animation_func(i,orders):
        # update line data
        for n, order in enumerate(orders):
            ok = (wavelengths[n]>2000) & (wavelengths[n]<8000)
            spec_lines[n][0].set_data([wavelengths[n][ok],spectra[n,i,ok]])
            ax[n].set_title("{} order, frame {}".format(order,i))

        return spec_lines
        
    # create and plot animation
    animation = FuncAnimation(fig, animation_func, init_func = init, frames = np.shape(spectra)[1], interval = 20,
                              fargs=(orders,))
    plt.tight_layout()

    # save animation
    if save_plot:
        stagedir = os.path.join(output_dir, f'stage{stage}/plots')

        if not os.path.exists(stagedir):
            os.makedirs(stagedir)

        animation.save(os.path.join(stagedir, '{}.gif'.format(filename)), writer = 'ffmpeg', fps = 10)

    if show_plot:
        plt.show(block = True)

    plt.close() # save memory
