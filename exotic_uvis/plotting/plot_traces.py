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
                    order="+1", show_plot = False, save_plot = False, 
                    stage = 0, output_dir = None):
    """Plot the fitted trace profiles.

    Args:
        y_vals (array-like): y positions of the fits.
        profile (array-like): pulled profiles as a function of y.
        gaussian_fit (array-like): fitted profiles as a function of y.
        cal_center (float): calculated center of the pulled profiles.
        fit_center (float): fitted center of the fitted profiles.
        show_plot (bool, optional): whether to show this plot.
        Defaults to False.
        save_plot (bool, optional): whether to save this plot.
        Defaults to False.
        stage (int, optional): stage this is being run in, for
        plot saving. Defaults to 0.
        output_dir (str, optional): where this plot is being saved to,
        if save_plot is True. Defaults to None.
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

    if save_plot:
        stagedir = os.path.join(output_dir, f'stage{stage}/plots/')
        if not os.path.exists(stagedir):
            os.makedirs(stagedir) 
        filedir = os.path.join(stagedir, 'trace_profile_order{}.png'.format(order))
        plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)

    if show_plot:
        plt.show(block=True)

    plt.close() # save memory
    
    return


def plot_fitted_positions():
    return

def plot_fitted_amplitudes():
    return

def plot_fitted_widths():
    return