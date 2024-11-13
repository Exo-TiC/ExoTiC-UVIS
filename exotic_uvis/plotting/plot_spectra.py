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
    """Function to plot one extracted spectrum.

    Args:
        wavelengths (np.array): wavelength solution for given order.
        spectrum (np.array): 1D extracted spectrum.
        order (str, optional): which order this is, for plot naming.
        Defaults to "+1".
        show_plot (bool, optional): whether to interrupt execution to
        show the user the plot. Defaults to False.
        save_plot (bool, optional): whether to save this plot to a file.
        Defaults to False.
        filename (str, optional): name to give this file, if saving.
        Defaults to None.
        output_dir (str, optional): where to save the file, if saving.
        Defaults to None.
    """

    # bound wavelengths to the region G280 is sensitive to
    ok = (wavelengths>2000) & (wavelengths<8000)

    # initialize plot and plot data that's in the okay range
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

    return 


def plot_spec_gif(wavelengths, spectra, orders=("+1",),
                  stage = 2, show_plot = False, save_plot = False,
                  filename = None, output_dir = None):
    """Plots gifs of the extracted spectra over time.

    Args:
        wavelengths (np.array): wavelength solution for given orders.
        spectra (np.array): 1D extracted spectra.
        order (str, optional): which orders we have, for plot naming.
        Defaults to ("+1",).
        show_plot (bool, optional): whether to interrupt execution to
        show the user the plot. Defaults to False.
        save_plot (bool, optional): whether to save this plot to a file.
        Defaults to False.
        filename (str, optional): name to give this file, if saving.
        Defaults to None.
        output_dir (str, optional): where to save the file, if saving.
        Defaults to None.
    """

    # define order colors
    colors = {"+1":'red',"-1":'blue',
              "+2":'orangered',"-2":'royalblue',
              "+3":'darkorange',"-3":'dodgerblue',
              "+4":'orange',"-4":'deepskyblue'}

    # create animation for each order
    for wav, spec, order in zip(wavelengths, spectra, orders):
        fig,ax = plt.subplots(figsize=(6,4))
        
        # plot first spectrum to get things started
        ok = (wav>2000) & (wav<8000)
        spec_line = ax.plot(wav[ok],spec[0,ok],color = colors[order],
                            label="{} order, frame 0".format(order))
        leg = ax.legend(loc='upper right')
        ax.set_xlim(2000,8000)
        ax.set_ylim(0, np.nanmax(spec[:,ok]))
        ax.set_xlabel('wavelength [AA]')
        ax.set_ylabel('counts [a.u.]')

        # initialize 
        def init():
            ok = (wav>2000) & (wav<8000)
            spec_line[0].set_data([wav[ok],spec[0,ok]])
            leg.get_texts()[0].set_text("{} order, frame {}".format(order,0))

            return spec_line

        # define animation function
        def animation_func(i):
            # update line data
            ok = (wav>2000) & (wav<8000)
            spec_line[0].set_data([wav[ok],spec[i,ok]])
            leg.get_texts()[0].set_text("{} order, frame {}".format(order,i))

            return spec_line
            
        # create and plot animation
        animation = FuncAnimation(fig, animation_func, init_func = init,
                                  frames = np.shape(spec)[0], interval = 20)
        plt.tight_layout()

        # save animation
        if save_plot:
            stagedir = os.path.join(output_dir, f'stage{stage}/plots')

            if not os.path.exists(stagedir):
                os.makedirs(stagedir)

            animation.save(os.path.join(stagedir, '{}_order.gif'.format(filename,order)), writer = 'ffmpeg', fps = 10)

        if show_plot:
            plt.show(block = True)

        plt.close() # save memory

    return 
