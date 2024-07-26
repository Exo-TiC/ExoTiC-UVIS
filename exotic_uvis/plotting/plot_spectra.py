import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr
import os
from exotic_uvis.plotting import plot_exposure

#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})



def plot_one_spectrum(wavelengths, spectrum, show_plot = False, 
                      save_plot = False, stage = 0, 
                      filename = None, output_dir = None):
    """Function to plot one extracted spectrum

    Args:
        wavelengths (_type_): _description_
        spectrum (_type_): _description_
        show_plot (bool, optional): _description_. Defaults to False.
        save_plot (bool, optional): _description_. Defaults to False.
        stage (int, optional): _description_. Defaults to 0.
        filename (_type_, optional): _description_. Defaults to None.
        output_dir (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    plt.figure(figsize = (10, 7))
    plt.plot(wavelengths, spectrum)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Extracted Counts')
    plt.title('Example of extracted spectrum')
    
    if show_plot:
        plt.show()

    if save_plot:
        plt.show()

    return 0