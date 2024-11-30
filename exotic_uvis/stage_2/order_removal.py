import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm
from matplotlib.pyplot import rc

rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


from exotic_uvis.plotting import plot_exposure


def Gauss1D(x, A, x0, sigma):
    """Creates a 1D Gaussian on the given x range.

    Args:
        x (np.array): independent variable in the Gaussian.
        H (float): vertical offset.
        A (float): amplitude of the Gaussian.
        x0 (float): center of the Gaussian.
        sigma (float): width of the Gaussian.

    Returns:
        np.array: Gaussian profile on domain x.
    """
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def exponential(x, x0, y0, a, b):

    """

    Function to define an inverse profile

    """


    return y0 + a * np.exp(b * (x - x0)) 



def remove_zeroth_order(obs, mode = 'radial_profile', zero_pos = [1158, 300], rmin = 100, rmax = 300, rwidth = 3, fit_profile = False, 
               verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):


    """
    
    Function to simulate the background flux due to the zeroth order around the edges of the spectra

    """

    images = obs.images.data.copy()

    if mode == 'radial_profile':

        zero_bkg = np.zeros_like(images)

        xvals = np.arange(np.shape(images)[2])
        yvals = np.arange(np.shape(images)[1])

        xx, yy = np.meshgrid(xvals, yvals)
        rr = np.sqrt((xx - zero_pos[0])**2 + (yy - zero_pos[1])**2)
        tt = np.arctan2((yy - zero_pos[1]), (xx - zero_pos[0]))

        profiles = []
        r_vect = np.arange(rmin, rmax, rwidth)
        r_cents = (r_vect[:-1] + r_vect[1:])/2

        ang = np.pi/6
        mask_angle = ((tt > ang) & (tt < np.pi - ang)) | ((tt > ang - np.pi) & (tt < -ang))

        print(np.sum(mask_angle))

        hist_area = images[0].copy()
        hist_area[mask_angle & (rr < rmax) & (rr > rmin)] = 'nan'
  
        plot_exposure([hist_area], title = 'Area to build the histogram', 
                      show_plot=(show_plots>0), save_plot=(save_plots>0),
                      output_dir=output_dir, filename = ['0th_order_histogram'])
        
        # test plots:
        #hist_area = images[0].copy()
        #total_mask = mask_angle & (rr < rmax) & (rr > rmin)
        #hist_area[~total_mask] = 'nan'
        
        #plot_exposure([hist_area], min = 0, max = 2)

      
        for j, image in enumerate(tqdm(images, desc = 'Calculating 0th order background... Process:')):

            profile = []
        
            if j >= 0:

                for i in range(len(r_vect) - 1):
                
                    mask_radius = (rr >= r_vect[i]) & (rr < r_vect[i + 1]) 
                    mask = mask_radius & mask_angle & (image != 0) #& (image < 60)
                
                    hist, bin_edges = np.histogram(image[mask], bins = np.linspace(-10, 60, 100))
                    bin_cents = (bin_edges[:-1] + bin_edges[1:])/2

                    parameters, covariance = optimize.curve_fit(Gauss1D, bin_cents, 
                                                                hist, 
                                                                p0 = [np.amax(hist), bin_cents[np.argmax(hist)], 10])
                

                    if i == int(len(r_vect)/2):
                        hist_area = image.copy()
                        hist_area[mask_angle & (rr < r_vect[i + 1]) & (rr > r_vect[i])] = 'nan'
  
                        plot_exposure([hist_area], title = 'Area to build the histogram', 
                                    show_plot=(show_plots>0), save_plot=(save_plots>0),
                                    output_dir=output_dir, filename = ['0th_order_histogram'])
                        
                        plt.figure(figsize = (10, 7))
                        plt.hist(image[mask], bins = np.linspace(-10, 60, 100), color = 'indianred', alpha = 0.7)
                        plt.plot(bin_cents, Gauss1D(bin_cents, parameters[0], parameters[1], parameters[2]), color='gray')
                        #plt.axvline(bin_edges[np.argmax(hist)], color = 'gray', linestyle = '--')
                        plt.axvline(np.median(image[mask]), color= 'gray')
                        #plt.axvline(np.median(image[mask]) + np.sqrt(np.median(image[mask])))
                        #plt.axvline(np.median(image[mask]) - np.sqrt(np.median(image[mask])))
                        plt.xlabel('Photons')
                        plt.ylabel('Counts')
                        plt.show(block=True)

                    profile.append(parameters[1]) # check, why negative values in histogram

                    if fit_profile == False:
                        image[mask_radius] -= parameters[1]
                        zero_bkg[j, mask_radius] = parameters[1]

                if fit_profile:

                    profile = np.array(profile)
                    mask_radius = (rr >= rmin) & (rr < rmax) 
                    mask = r_cents > 0
                    popt, pcov = optimize.curve_fit(exponential, r_cents[mask], profile[mask], p0 = [100., 7., 1., -0.1])
                    
                    fitted_profile = exponential(r_cents, popt[0], popt[1], popt[2], popt[3])
                    image[mask_radius] -= exponential(rr[mask_radius], popt[0], popt[1], popt[2], popt[3])
                    zero_bkg[j, mask_radius] = exponential(rr[mask_radius], popt[0], popt[1], popt[2], popt[3])

                    plt.figure(figsize=(10, 7))
                    plt.plot(r_cents, profile, '-o', color='indianred')
                    if fit_profile:
                        plt.plot(r_cents, fitted_profile, color = 'gray')
                    plt.xlabel('Distance from 0th order')
                    plt.ylabel('0th order value')
                    plt.show()


                    
                profiles.append(profile)
                
                if (j == 0) and ((show_plots>0) or (save_plots>0)):
                    #utils.plot_image([obs.images.data[j, 1000:1400, 1700:2600], image[1000:1400, 1700:2600]], min = -1)
                    #utils.plot_image([obs.images.data[j], image], min = -1)
                    #utils.plot_image([zero_bkg[j]], min = 0.95, max = 1.8)

                    plot_exposure([obs.images.data[j, :], image], title = '0th order removal example', 
                                    show_plot=(show_plots>0), save_plot=(save_plots>0),
                                    output_dir=output_dir, filename = ['0th_order_histogram'])
                    
                    plot_exposure([zero_bkg[j]], title = '0th order model', min=np.min(fitted_profile), max=np.max(fitted_profile),
                                    show_plot=(show_plots>0), save_plot=(save_plots>0),
                                    output_dir=output_dir, filename = ['0th_order_histogram'])
        

                    
        profiles = np.array(profiles)
        obs.images.data = images

    return zero_bkg