import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def plot_one_spectrum(wavelengths, spectrum, order="+1",
                      show_plot = False, save_plot = False,
                      filename = None, output_dir = None):
    """Function to plot one extracted spectrum.

    Args:
        wavelengths (np.array): wavelength solution for given order.
        spectrum (np.array): 1D extracted spectrum.
        order (str, optional): which order this is, for plot title.
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
    # define order colors
    colors = {"+1":'indianred',"-1":'dodgerblue',
              "+2":'orangered',"-2":'royalblue',
              "+3":'darkorange',"-3":'blue',
              "+4":'orange',"-4":'deepskyblue'}

    # bound wavelengths to the region G280 is sensitive to
    ok = (wavelengths>2000) & (wavelengths<8000)

    # initialize plot and plot data that's in the okay range
    plt.figure(figsize = (10, 7))
    plt.plot(wavelengths[ok], spectrum[ok], color=colors[order])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Extracted Counts')
    plt.title('Example of extracted order {} spectrum'.format(order))
    
    if save_plot:
        plot_dir = os.path.join(output_dir, 'plots') 
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir) 
        filedir = os.path.join(plot_dir, f'{filename}.png')
        plt.savefig(filedir,dpi=300,bbox_inches='tight')

    if show_plot:
        plt.show(block=True)

    plt.close() # save memory

    return 


def plot_spec_gif(wav, spec, order="+1",
                  show_plot = False, save_plot = False,
                  filename = None, output_dir = None):
    """Plots gifs of the extracted spectra over time.

    Args:
        wav (np.array): wavelength solution for given orders.
        spec (np.array): 1D extracted spectra.
        order (str, optional): which order we are plotting, for plot title.
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

    # define order colors
    colors = {"+1":'indianred',"-1":'dodgerblue',
              "+2":'orangered',"-2":'royalblue',
              "+3":'darkorange',"-3":'blue',
              "+4":'orange',"-4":'deepskyblue'}

    # create animation for each order
    #for wav, spec, order in zip(wavelengths, spectra, orders):
    fig,ax = plt.subplots(figsize=(10, 7))
    
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
        plot_dir = os.path.join(output_dir, 'plots') 

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        animation.save(os.path.join(plot_dir, f'{filename}.gif'), writer = 'ffmpeg', fps = 10)

    if show_plot:
        plt.show(block = True)

    plt.close() # save memory

    return 


def plot_2d_spectra(wav, spec, order="+1", 
                    show_plot = False, save_plot = False,
                    filename = None, output_dir = None):
    """Plots 2D image of the normalised spectra over time.

    Args:
        wav (np.array): wavelength solution for given orders.
        spec (np.array): 1D extracted spectra.
        show_plot (bool, optional): whether to interrupt execution to
        show the user the plot. Defaults to False.
        save_plot (bool, optional): whether to save this plot to a file.
        Defaults to False.
        filename (str, optional): name to give this file, if saving.
        Defaults to None.
        output_dir (str, optional): where to save the file, if saving.
        Defaults to None.
    """
    
    # normalize spectra
    #print(np.shape(np.nanmedian(spec[:20], axis=0)))
    n_oot = int(0.20*spec.shape[0]) # typically, first 20% of data is the first orbit, which is oot/ooe
    spec = spec / np.nanmedian(spec[:n_oot], axis=0)

    
    plt.figure(figsize = (10, 7))
    plt.imshow(spec, aspect='auto', origin='lower',
               vmin = 0.99, vmax = 1.01, cmap='copper',
               extent = [wav[0], wav[-1], 0, spec.shape[0]])
    plt.colorbar()
    plt.ylabel('Integration number')
    plt.xlabel('Wavelength (microns)')
    plt.title(f'2D spectral map Order {order}')

    if save_plot:
        plot_dir = os.path.join(output_dir, 'plots') 
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir) 
        filedir = os.path.join(plot_dir, f'{filename}.png')
        plt.savefig(filedir,dpi=300,bbox_inches='tight')

    if show_plot:
        plt.show(block=True)

    plt.close() # save memory

    return 
