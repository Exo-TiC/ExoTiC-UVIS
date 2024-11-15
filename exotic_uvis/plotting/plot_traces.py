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




def plot_profile_fit(y_vals, profile, gaussian_fit, cal_center, fit_center,
                    show_plot = False, save_plot = False, 
                    stage = 0, filename = None, output_dir = None):
    """_summary_

    Args:
        y_vals (_type_): _description_
        profile (_type_): _description_
        gaussian_fit (_type_): _description_
        cal_center (_type_): _description_
        fit_center (_type_): _description_
        show_plot (bool, optional): _description_. Defaults to False.
        save_plot (bool, optional): _description_. Defaults to False.
        stage (int, optional): _description_. Defaults to 0.
        filename (_type_, optional): _description_. Defaults to None.
        output_dir (_type_, optional): _description_. Defaults to None.
    """

    plt.figure(figsize = (10, 7))
    plt.plot(y_vals, profile, color = 'indianred')
    plt.plot(y_vals, gaussian_fit, linestyle = '--', linewidth = 1.2, color = 'gray')
    plt.axvline(fit_center, linestyle = '--', color = 'gray', linewidth = 0.7)
    plt.axvline(fit_center - 12, linestyle = '--', color = 'gray', linewidth = 0.7)
    plt.axvline(fit_center + 12, linestyle = '--', color = 'gray', linewidth = 0.7)
    plt.axvline(cal_center, color = 'black', linestyle = '-.', alpha = 0.8)
    plt.ylabel('Counts')
    plt.xlabel('Detector Pixel Position')
    plt.title('Example of Profile fitted to Trace')


    plt.show(block=True)



    return




def plot_fitted_positions():
    return

def plot_fitted_amplitudes():
    return

def plot_fitted_widths():
    return