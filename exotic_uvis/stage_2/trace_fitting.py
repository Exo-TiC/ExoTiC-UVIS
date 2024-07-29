import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import grismconf
from tqdm import tqdm

from exotic_uvis.plotting import plot_exposure

def get_trace_solution(obs, order, source_pos, refine_calibration, path_to_cal):
    '''
    Pulls the region of each image that has the trace in it, with wavelength solution provided by GRISMCONF.

    :param obs: xarray. Its obs.images DataSet contains the images and its obs.subarr_coords DataSet is used to trick the configuration into thinking it is embedded.
    :param order: str. Options are "+1", "-1", "+2", "-2", etc. Which order you want to pull.
    :param source_pos: tuple of float. The x, y position of the source in the unembedded direct image.
    :param refine_calibration: bool. If True, uses Gaussian fitting to improve the location of the trace.
    :param path_to_cal: str. Path to the GRISMCONF calibration file used to locate the trace.
    :return: the trace x and y positions from the direct image, with the wavelength solution and optional sensitivity corrections.
    '''
    # Use the order and calibration software information to get the x range.
    dxs = get_x_range(order)

    # Get the mean subarr_coords offsets.
    offset_x0 = np.mean(obs.subarr_coords.values[:,0])
    offset_y0 = np.mean(obs.subarr_coords.values[:,2])
    adjusted_x0 = source_pos[0] + offset_x0
    adjusted_y0 = source_pos[1] + offset_y0

    # Get the x, y positions of the trace as well as wavelength solution and sensitivity correction.
    trace_x, trace_y, wavs, fs = get_calibration_trace(order,
                                                       adjusted_x0,
                                                       adjusted_y0,
                                                       dxs,
                                                       path_to_cal)

    # Undo the offsets from the subarray coordinates.
    trace_x = [i - offset_x0 for i in trace_x]
    trace_y = [i - offset_y0 for i in trace_y]

    # Convert to numpy arrays.
    trace_x = np.array(trace_x)
    trace_y = np.array(trace_y)
    wavs = np.array(wavs)
    fs = np.array(fs)

    # test plot
    plot_exposure([obs.images.data[0]], line_data=[[trace_x, trace_y]], filename = 'None', save_plot=0, show_plot=2, output_dir='None')
    
    # Use Gaussian fitting to refine the y positions if asked.
    if refine_calibration:
        trace_y, widths = fit_trace(obs, trace_x, trace_y, profile_width = 70, pol_deg = 7, fit_type = 'Gaussian',
                                    fit_trace = False, plot_profile = None, check_all = False)
        
        # test plot
        plot_exposure([obs.images.data[0]], line_data=[[trace_x, trace_y[0]]], filename = 'None', save_plot=0, show_plot=2, output_dir='None')
    
    else:
        # No information obtained about trace widths.
        refined_trace_y = np.empty((obs.images.shape[0],trace_x.shape[0]))
        for k in range(obs.images.shape[0]):
            refined_trace_y[k,:] = trace_y
        trace_y = refined_trace_y
        widths = None
    return trace_x, trace_y, wavs, widths, fs

def get_x_range(order):
    '''
    Gets the x range of the order requested.

    :param order: str. Options are "+1", "-1", "+2", "-2", etc. Which order you want to pull.
    :return: the x range and the order-to-letter/symbol translator dictionary.
    '''
    if order == "+1":
        dxs = (-550, 0)
    
    if order == "-1":
        dxs = (225, 775)
    
    if order == "+2":
        dxs = (-700, -150) # is this right?

    if order == "-2":
        dxs = (225+150, 225+150+550) # is this right?
    
    if order == "+3":
        dxs = (-950, -300) # is this right?
    
    if order == "-3":
        dxs = (625, 625+550) # is this right?

    if order == "+4":
        dxs = (-1300, -1300+550) # is this right?
    
    if order == "-4":
        dxs = (875, 875+550) # is this right?

    return dxs

def get_calibration_trace(order,x0,y0,dxs,path_to_cal):
    '''
    Uses the supplied calibration software and source position to locate the trace and assign wavelength solution.

    :param order: str. Options are "+1", "-1", "+2", "-2", etc. Used to grab the right calibration from the calibration file.
    :param x0: float. Embedded x position of the source.
    :param y0: float. Embedded y position of the source.
    :param dxs: tuple of int. The x range spanned by the trace.
    :param path_to_cal: str. Path to the calibration file used to locate the trace.
    :return: the x and y positions of the calibrated trace, the assigned wavelength solution, and the sensitivity corrections.
    '''
    # Initialize the GRISMCONF configuration.
    C = grismconf.Config(path_to_cal)

    # Turn the dxs limits into a full span of column positions.
    dxs = np.arange(dxs[0],dxs[1],1)

    # Compute the t values corresponding to the exact offsets
    ts = C.INVDISPX(order,x0,y0,dxs)
    # Compute the dys values for the same pixels
    dys = C.DISPY(order,x0,y0,ts)
    # Compute wavelength of each of the pixels
    wavs = C.DISPL(order,x0,y0,ts)
    
    # Combine the displacements with the source position to get the trace location.
    xs = [i+x0 for i in dxs]
    ys = [i+y0 for i in dys]

    # Get the sensitivity correction.
    s = C.SENS[order]
    fs = s.f

    return xs, ys, wavs, fs

def Gauss1D(x, H, A, x0, sigma):
    '''
    Plots a 1D Gaussian on the given x range.

    :param x: independent variable in the Gaussian.
    :param H: vertical offset.
    :param A: amplitude of the Gaussian.
    :param x0: center of the Gaussian.
    :param sigma: width of the Gaussian.
    :return: np.array of Gaussian profile fitted to x.
    '''
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fit_trace(obs, trace_x, trace_y, 
              profile_width = 70, pol_deg = 7, fit_type = 'Gaussian',
              fit_trace = False, plot_profile = None, check_all = False):
    '''
    Refines the trace vertical location by fitting profile curves to the cross-dispersion profiles.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param trace_x: lst of floats. The calibrated x positions of the trace pixels.
    :param trace_y: lst of floats. The calibrated y positions of the trace pixels.
    :param profile_width: float. How far up and down from the calibrated y positions to fit the Gaussian profiles.
    :param pol_deg: int. The degree of polynomial to fit to the trace position, if fit_trace is True.
    :param fit_type: str. Only choicee is 'Gaussian'. Type of profile to fit to the cross-dispersion profiles.
    :param fit_trace: bool. If True, fit a polynomial to the refined x, y positions of the trace in each frame.
    :param plot_profile: None.
    :param check_all: bool.
    :return: refined vertical positions of the traces and widths of the fitted profiles.
    '''
    # Initialize traces and widths.
    traces, widths = [], []

    # Copy image data and extract y values.
    images = obs.images.data.copy()
    y_data = range(obs.dims['y'])

    # Iterate over all images.
    for i, image in enumerate(tqdm(images, desc = 'Computing trace... Progress:')):
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
                plt.show()

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
