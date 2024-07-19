from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from photutils import DAOStarFinder

import numpy as np
import matplotlib.pyplot as plt

def locate_target(direct_image):
    '''
    Opens the direct image and finds the target star.
    
    :param direct_image: str. Path to the direct image.
    :return: location of the direct image in x, y floats.
    '''    
    # Define three parameters for finding the target.
    satisfied = False       # whether the user is happy with the location identified by DAOStarFinder
    search_radius = 100     # how far to search from the initial guess
    threshold = 50          # how many sigma from the frame median your source is, which is quite a few
    with fits.open(direct_image) as fits_file:
        # Open the data and show it to the user.
        d = fits_file[1].data
        plt.imshow(d, vmin=0, vmax=100, origin='lower', cmap='binary_r')
        plt.title("Direct image for target finding")
        plt.show()
        plt.close()
        while not satisfied:
            mean, median, std = sigma_clipped_stats(d, sigma=3.0) 
            daofind = DAOStarFinder(fwhm=3.0, threshold=threshold*std)    

            sources = daofind(d - median)
            for col in sources.colnames:    
                sources[col].info.format = '%.8g'  # for consistent table output

            # Ask the user to identify the star.
            bestx = int(input("Please input your best guess for the x position of the source star: "))
            besty = int(input("Please input your best guess for the y position of the source star: "))

            possible_sources = []

            for ind1, ind2 in zip(sources[1][:], sources[2][:]):
                if (((ind1-bestx)**2 + (ind2-besty)**2)**0.5 < search_radius):
                    possible_sources.append((ind1, ind2))
            print("Located %.0f possible sources." % len(possible_sources))
            print("Please select the source from this list:")
            for ind, item in enumerate(possible_sources):
                print(str(ind) + "     " + str(item))
            ind = int(input(""))
            xs, ys = possible_sources[ind]

            plt.subplot(1,1,1)
            plt.imshow(d, vmin=0, vmax=100, origin='lower', cmap='binary_r')
            plt.scatter(xs, ys, s=8, marker='x', color='red')
            plt.xlim(int(xs-70), int(xs+70))
            plt.ylim(int(ys-70), int(ys+70))

            plt.show()
            plt.close()

            check = int(input("Enter 0 to keep this source, 1 to search again,\n2 to update the threshold/search radius and then search again, or 3 to select manually: "))
            if check == 0:
                satisfied = True
            if check == 2:
                print("Current threshold: %.3f" % threshold)
                threshold = int(input("Enter new threshold: "))
                print("Current search radius: %.3f" % search_radius)
                search_radius = int(input("Enter new search radius: "))
            if check == 3:
                xs = float(input("Enter x: "))
                ys = float(input("Enter y: "))
                plt.subplot(1,1,1)
                plt.imshow(d, vmin=0, vmax=100, origin='lower', cmap='binary_r')
                plt.scatter(xs, ys, s=8, marker="x", color="red")
                plt.xlim(int(xs-70), int(xs+70))
                plt.ylim(int(ys-70), int(ys+70))
                plt.show()
                plt.close()
                check2 = int(input("Is this okay (0) or do you want to search again (1)?: "))
                if check2 == 0:
                    satisfied = True
    print("Source selected:", xs, ys)
    return xs, ys