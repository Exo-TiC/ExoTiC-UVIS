import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy import optimize
from photutils.centroids import centroid_com, centroid_2dg
import grismconf
from tqdm import tqdm
from astropy.modeling.models import Moffat1D
from scipy.special import voigt_profile


from exotic_uvis.plotting import plot_exposure



def config_order_to_parameters(order, config):

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

    if config == "aXe":
        letters = {"+1":"A",
                   "-1":"B",
                   "+2":"C",
                   "-2":"D",
                   "+3":"E",
                   "-3":"F",
                   "+4":"G",
                   "-4":"H"}

    if config == "GRISM":
        letters = {"+1":"+1",
                   "-1":"-1",
                   "+2":"+2",
                   "-2":"-2",
                   "+3":"+3",
                   "-3":"-3",
                   "+4":"+4",
                   "-4":"-4"}
    
    return dxs, letters


def get_calibration_trace(letter,x0,y0,dxs,path_to_config):

    C = grismconf.Config(path_to_config)

    # Need to get dxs to find ts and then dys.
    dxs = np.arange(dxs[0],dxs[1],1)

    # Compute the t values corresponding to the exact offsets
    ts = C.INVDISPX(letter,x0,y0,dxs)
    # Compute the dys values for the same pixels
    dys = C.DISPY(letter,x0,y0,ts)
    # Compute wavelength of each of the pixels
    wavs = C.DISPL(letter,x0,y0,ts)
    
    xs = [i+x0 for i in dxs]
    ys = [i+y0 for i in dys]

    s = C.SENS[letter]
    fs = s.f

    return xs, ys, wavs, fs



def Gauss1D(x, H, A, x0, sigma):

    """

    Function to return a 1D Gaussian profile 

    """

    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fit_trace(obs, trace_x, trace_y, 
              profile_width = 70, pol_deg = 7, fit_type = 'Gaussian',
              fit_trace = False, plot_profile = None, check_all = False):


    """
    
    Function to find the trace by fitting a Gaussian curve to the cross-dispersion profiles
    
    """

    # initialize traces and widths
    traces, widths = [], []

    # copy image data and extract y values
    images = obs.images.data.copy()
    y_data = range(obs.dims['y'])

    # iterate over all images
    for i, image in enumerate(tqdm(images, desc = 'Computing trace... Progress:')):

        #initialize image trace and width
        trace, width = [], []

        # iterate over all pixels in trace
        for j, pix in enumerate(trace_x):

            # get center from calibrated trace and define profile to fit
            center = int(trace_y[j])
            low_val, up_val = center - profile_width, center + profile_width
            profile = image[low_val: up_val, int(pix)]
            y_vals = y_data[low_val: up_val]
            
            # fit a Gaussian profile
            if fit_type == 'Gaussian':
                parameters, covariance = optimize.curve_fit(Gauss1D, 
                                                            y_vals, 
                                                            profile, 
                                                            p0 = [0, np.amax(profile), 
                                                            y_vals[np.argmax(profile)], 1])

            # append trace and fwhm
            trace.append(parameters[2])
            width.append(2*np.sqrt(2*np.log(2)) * parameters[3])

    
            # plot the j profile in the i image
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


        # if true, fit a polynomial to the extracted trace locations and widths
        if fit_trace:
            
            # fit trace centers, improve this fitting: shift old polynomial with coefficients
            coeffs = np.polyfit(trace_x, trace, deg = pol_deg)
            trace = np.polyval(coeffs, trace_x)

            # fit trace widths
            coeffs = np.polyfit(trace_x, width, deg = pol_deg)
            width = np.polyval(coeffs, trace_x)
        

        # if true, plot all the traces over the image for comparison/validation
        if check_all:
            plot_exposure([image], line_data = [[trace_x, trace_y], [trace_x, trace]], min = 0)

        # append
        traces.append(trace)
        widths.append(width)

    return np.array(traces), np.array(widths)
