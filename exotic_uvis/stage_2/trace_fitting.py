import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import grismconf
from tqdm import tqdm

from exotic_uvis.plotting import plot_exposure

def get_trace_solution(obs, order, source_pos, refine_calibration, path_to_cal,
                       verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    """Pulls the region of each image that has the trace in it, with wavelength
    solution provided by GRISMCONF.

    Args:
        obs (xarray): obs.images DataSet contains the images and obs.subarr_coords
        DataSet is used to trick the configuration into thinking it is embedded.
        order (str): options are "+1", "-1", "+2", "-2", etc. Which order you
        want to pull.
        source_pos (tup): x, y float position of the source in the unembedded
        direct image.
        refine_calibration (bool): if True, uses Gaussian fitting to improve
        the location of the trace.
        path_to_cal (str): path to the GRISMCONF calibration file used to
        locate the trace.
        verbose (int, optional): How detailed you want the printed statements
        to be. Defaults to 0.
        show_plots (int, optional): How many plots you want to show. Defaults to 0.
        save_plots (int, optional): How many plots you want to save. Defaults to 0.
        output_dir (str, optional): Where to save the plots to, if save_plots
        is greater than 0. Defaults to None.

    Returns:
        np.array,np.array,np.array,np.array,scipy.interpolate: trace x and y
        positions, wavelength solutions, optional trace widths, and optional
        sensitivity correction functions.
    """
   
    # Get the mean subarr_coords offsets.
    offset_x0 = obs.subarr_coords.values[0]
    offset_y0 = obs.subarr_coords.values[2]
    adjusted_x0 = source_pos[0] + offset_x0
    adjusted_y0 = source_pos[1] + offset_y0

    # Get the x, y positions of the trace as well as wavelength solution and sensitivity correction.
    trace_x, trace_y, wavs, sens = get_calibration_trace(order,
                                                         adjusted_x0,
                                                         adjusted_y0,
                                                         path_to_cal)

    # Undo the offsets from the subarray coordinates.
    trace_x = [i - offset_x0 for i in trace_x]
    trace_y = [i - offset_y0 for i in trace_y]

    # Convert to numpy arrays.
    trace_x = np.array(trace_x)
    trace_y = np.array(trace_y)
    wavs = np.array(wavs)
    sens = np.array(sens)

    # Plot the calibration over the image.
    if (show_plots > 0 or save_plots > 0):
        plot_exposure([obs.images.data[0]], line_data=[[trace_x, trace_y]],
                      filename = ['s2_calibration_{}'.format(order)], stage=2,
                      save_plot=(save_plots>0), show_plot=(show_plots>0),
                      output_dir=output_dir)
    
    # Use Gaussian fitting to refine the y positions if asked.
    if refine_calibration:
        trace_y, widths = fit_trace(obs, trace_x, trace_y, profile_width = 70, pol_deg = 7, fit_type = 'Gaussian',
                                    fit_trace = False, plot_profile = [20, 300], check_all = False, verbose = verbose)
        
        # Plot refined calibration.
        plot_exposure([obs.images.data[0]], line_data=[[trace_x, trace_y[0]]],
                      filename = ['s2_refined-calibration_{}'.format(order)], stage=2,
                      save_plot=(save_plots>0), show_plot=(show_plots>0),
                      output_dir=output_dir)
    
    else:
        # No information obtained about trace widths.
        refined_trace_y = np.empty((obs.images.shape[0],trace_x.shape[0]))
        for k in range(obs.images.shape[0]):
            refined_trace_y[k,:] = trace_y
        trace_y = refined_trace_y
        widths = None
    return trace_x, trace_y, wavs, widths, sens

def get_calibration_trace(obs, order, x0, y0, path_to_cal):
    """Uses the supplied calibration software and source position to locate the
    trace and assign wavelength solution.

    Args:
        order (str): Options are "+1", "-1", "+2", "-2", etc. Used to grab the
        right calibration from the calibration file.
        x0 (float): Embedded x position of the source.
        y0 (float): Embedded y position of the source.
        path_to_cal (str): Path to the calibration file used to locate the trace.

    Returns:
        list,list,np.array,scipy.interpolate: the x and y positions of the calibrated
        trace, the assigned wavelength solution, and the sensitivity correction function.
    """
    # Initialize the GRISMCONF configuration.
    C = grismconf.Config(path_to_cal)

    # Get dx limits using 0<t<1.
    dxs = C.DISPX(order,x0,y0,np.array([0,1]))
    dxs = np.sort(dxs)

    # Turn the dxs limits into a full span of column positions.
    dxs = np.arange(dxs[0],dxs[1],1)

    # extract x values relative to x00 (need to change y to x)
    #xvals_rel = np.arange(obs.dims['y']) - x0

    # Compute the t values corresponding to the exact offsets
    ts = C.INVDISPX(order,x0,y0,dxs)

    # Compute the dys values for the same pixels
    dys = C.DISPY(order,x0,y0,ts)

    # Compute wavelength of each of the pixels
    wavs = C.DISPL(order,x0,y0,ts)

    # Restrict attention to just where 200 nm < wavs < 800 nm.
    dxs = dxs[np.logical_and(wavs>=2000, wavs<=8000)]
    dys = dys[np.logical_and(wavs>=2000, wavs<=8000)]
    wavs = wavs[np.logical_and(wavs>=2000, wavs<=8000)]
    
    # Combine the displacements with the source position to get the trace location.
    xs = [i+x0 for i in dxs]
    ys = [i+y0 for i in dys]

    # Get the sensitivity correction for the wavelength range we are working on.
    s = C.SENS[order]
    fs = s.f
    sens = fs(wavs)

    return xs, ys, wavs, sens

def Gauss1D(x, H, A, x0, sigma):
    """Plots a 1D Gaussian on the given x range.

    Args:
        x (np.array): independent variable in the Gaussian.
        H (float): vertical offset.
        A (float): amplitude of the Gaussian.
        x0 (float): center of the Gaussian.
        sigma (float): width of the Gaussian.

    Returns:
        np.array: Gaussian profile on domain x.
    """
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fit_trace(obs, trace_x, trace_y, 
              profile_width = 40, pol_deg = 7, fit_type = 'Gaussian',
              fit_trace = False, plot_profile = None, check_all = False, verbose = 0):
    """Refines the trace vertical location by fitting profile curves to the
    cross-dispersion profiles.

    Args:
        obs (xarray): obs.images contains the images.
        trace_x (np.array): calibrated x positions of trace pixels.
        trace_y (np.array): calibrated y positions of trace pixels.
        profile_width (int, optional): how far up and down to fit the trace
        profile. Defaults to 40.
        pol_deg (int, optional): degree of polynomial to fit to the trace
        position, if fit_trace is True. Defaults to 7.
        fit_type (str, optional): Actually has to be Gaussian? Defaults to
        'Gaussian'.
        fit_trace (bool, optional): if True, fit a polynomial to the refined
        x, y positions of the trace in each frame. Defaults to False.
        plot_profile (_type_, optional): _description_. Defaults to None.
        check_all (bool, optional): _description_. Defaults to False.

    Returns:
        np.array,np.array: refined trace positions and widths.
    """
    # Initialize traces and widths.
    traces, widths = [], []

    # Copy image data and extract y values.
    images = obs.images.data.copy()
    y_data = range(obs.dims['y'])

    # Iterate over all images.
    for i, image in enumerate(tqdm(images, desc = 'Computing trace... Progress:',
                                   disable=(verbose<1))):
        # Initialize trace y positions and profile widths and for this image.
        trace, width = [], []

        # Iterate over every column to refine the y position.
        for j, pix in enumerate(trace_x):
            # Get center from calibrated trace and define profile to fit.
            center = int(trace_y[j])
            low_val, up_val = center - profile_width, center + profile_width
            profile = image[low_val: up_val, int(pix)] # profile runs from y0 - w to y0 + w
            y_vals = y_data[low_val: up_val] # the independent variable in the Gaussian fit
            
            # Fit a Gaussian profile.
            if fit_type == 'Gaussian':
                parameters, covariance = optimize.curve_fit(Gauss1D, 
                                                            y_vals, 
                                                            profile, 
                                                            p0 = [0, np.amax(profile), 
                                                            y_vals[np.argmax(profile)], 1])

            # Append refined y0 position and fwhm of the profile.
            trace.append(parameters[2])
            width.append(2*np.sqrt(2*np.log(2)) * parameters[3])
    
            # Plot the j profile in the i image.
            if (int(plot_profile[0]) == i) and (int(plot_profile[1]) == j): 
                plt.figure(figsize = (10, 7))
                plt.plot(y_vals, profile, color = 'indianred')
                plt.plot(y_vals, Gauss1D(y_vals, parameters[0], parameters[1], parameters[2], parameters[3]), linestyle = '--', linewidth = 1.2, color = 'gray')
                plt.axvline(parameters[2], linestyle = '--', color = 'gray', linewidth = 0.7)
                plt.axvline(parameters[2] - 12, linestyle = '--', color = 'gray', linewidth = 0.7)
                plt.axvline(parameters[2] + 12, linestyle = '--', color = 'gray', linewidth = 0.7)
                plt.axvline(trace_y[j], color = 'black', linestyle = '-.', alpha = 0.8)
                plt.ylabel('Counts')
                plt.xlabel('Detector Pixel Position')
                plt.title('Example of Profile fitted to Trace')
                #plt.savefig('PLOTS/profile.pdf', bbox_inches = 'tight')
                plt.show(block=True)

        # If true, fit a polynomial to the extracted trace locations and widths.
        if fit_trace:
            
            # Fit trace centers, improve this fitting: shift old polynomial with coefficients.
            coeffs = np.polyfit(trace_x, trace, deg = pol_deg)
            trace = np.polyval(coeffs, trace_x)

            # Fit trace widths.
            coeffs = np.polyfit(trace_x, width, deg = pol_deg)
            width = np.polyval(coeffs, trace_x)

        # If true, plot all the traces over the image for comparison/validation.
        if check_all:
            plot_exposure([image], line_data = [[trace_x, trace_y], [trace_x, trace]], min = 0)

        # Append this frame's y positions and dispersion profile widths to the entire set.
        traces.append(trace)
        widths.append(width)

    return np.array(traces), np.array(widths)
