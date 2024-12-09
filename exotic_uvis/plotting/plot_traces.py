import os

import matplotlib.pyplot as plt


#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def plot_profile_fit(y_vals, profile, gaussian_fit, cal_center, fit_center,
                    order="+1", column=0, show_plot = False, save_plot = False, 
                    output_dir = None):
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
    plt.title(f'Example of Profile fitted to Trace Column {column}')

    if save_plot:
        plot_dir = os.path.join(output_dir, 'plots') 
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir) 
        filedir = os.path.join(plot_dir, 'trace_profile_order{}_column{}.png'.format(order, column))
        plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)

    if show_plot:
        plt.show(block=True)

    plt.close() # save memory
    
    return


def plot_fitted_positions(trace_x, trace_y, trace, exp_num, fitted_trace = None, 
                          show_plot = False, save_plot = False, filename = None, output_dir = None):
    """Plots the GRISMCONF y-pos versus the fitted y-pos of the trace.

    Args:
        trace_x (array-like): GRISMCONF solution to the columns.
        trace_y (array-like): GRISMCONF solution to the rows.
        trace (array-like): _description_
        exp_num (int): which frame this is, for plot title and filename.
        fitted_trace (array-like, optional): a polynomial fit to the trace
        center, may or may not be performed. Defaults to None.
        show_plot (bool, optional): whether to interrupt execution to show the
        user the plot. Defaults to False.
        save_plot (bool, optional): whether to save this plot to a file.
        Defaults to False.
        filename (str, optional): name to give this file, if saving.
        Defaults to None.
        output_dir (str, optional): where to save the file, if saving.
        Defaults to None.
    """

    # plot the computed positions and compare to calibration trace
    plt.figure(figsize=(10, 7))
    plt.plot(trace_x, trace, 'o', alpha = 0.4, color='indianred', label='Profile centers')
    plt.plot(trace_x, trace_y, '--', color='gray', label='Calibration trace')

    if fitted_trace is not None:
        plt.plot(trace_x, fitted_trace, '-', color='black', label='Polynomial fit to centers')

    plt.xlabel('X pixel position')
    plt.ylabel('Y pixel position')
    plt.title(f'Trace positions Exposure {exp_num}')
    plt.legend()

    if save_plot:
        plot_dir = os.path.join(output_dir, 'plots') 
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir) 
        filedir = os.path.join(plot_dir, f'{filename}_frame{exp_num}.png')
        plt.savefig(filedir, bbox_inches = 'tight', dpi = 300)

    if show_plot:
        plt.show(block=True)

    plt.close() # save memory

    return


def plot_fitted_amplitudes():
    return

def plot_fitted_widths():
    return