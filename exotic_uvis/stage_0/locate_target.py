from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from photutils import DAOStarFinder

import numpy as np
import matplotlib.pyplot as plt


def locate_target(direct_image,test=False):
    """Uses the direct image and user feedback to locate the
    target, necessary for trace calibration and source
    tracking in later stages.

    Args:
        direct_image (str): Path to the direct image.
        test (bool, optional): Whether you are running in testing mode.
        Only used for validation on github. Defaults to False.

    Returns:
        float, float: Location of the direct image in x, y floats.
    """
    # Define three parameters for finding the target.
    satisfied = False       # whether the user is happy with the location identified by DAOStarFinder
    search_radius = 100     # how far to search from the initial guess
    threshold = 50          # how many sigma from the frame median your source is, which is quite a few
    fwhm = 3.0              # expected FWHM of a star

    # Not actually finding a source, just testing!
    if test:
        # Create a fake star in the center of the image.
        d = np.zeros((100,100))
        for i in range(100):
            for j in range(100):
                d[i,j] = np.exp(-((i-50)**2 + (j-50)**2)/3)
        
        # Use DAOStarFinder to locate possible targets.
        mean, median, std = sigma_clipped_stats(d, sigma=3.0) 
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)    

        sources = daofind(d - median)
        for col in sources.colnames:    
            sources[col].info.format = '%.8g'  # for consistent table output

        # Pull all DAOStarFinder sources to show the user what has been identified.
        xs = [x[1] for x in sources]
        ys = [y[2] for y in sources]

        # Offset best guess to ensure it knows how to return to the correct location.
        bestx, besty = (45,45)

        # Present options from the sources list that are close to the guess.
        possible_sources = []
        for ind1, ind2 in zip(xs,ys):
            if (((ind1-bestx)**2 + (ind2-besty)**2)**0.5 < search_radius):
                possible_sources.append((ind1, ind2))
        
        print("Located %.0f possible sources." % len(possible_sources))
        xs, ys = possible_sources[0]
    
    # Actually finding a file.
    else:
        with fits.open(direct_image) as fits_file:
            # Open the data so we can show it to the user.
            d = fits_file[1].data
            while not satisfied:
                # Use DAOStarFinder to locate possible targets.
                mean, median, std = sigma_clipped_stats(d, sigma=3.0) 
                daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)    

                sources = daofind(d - median)
                for col in sources.colnames:    
                    sources[col].info.format = '%.8g'  # for consistent table output

                # Pull all DAOStarFinder sources to show the user what has been identified.
                xs = [x[1] for x in sources]
                ys = [y[2] for y in sources]

                # Show the direct image. This supercedes any show_plots/save_plots call because it is mandatory for locating the image.
                plt.imshow(d, vmin=0, vmax=100, origin='lower', cmap='binary_r')
                plt.scatter(xs,ys,color='red',marker='*',alpha=0.5,label='DAOStarFinder sources',s=2)
                plt.title("Direct image for target finding")
                plt.legend()
                plt.show(block=True)
                plt.close()

                # Ask the user to guess where the star is.
                bestx = int(input("Please input your best guess for the x position of the source star: "))
                besty = int(input("Please input your best guess for the y position of the source star: "))

                # Present options from the sources list that are close to the guess.
                possible_sources = []

                for ind1, ind2 in zip(xs,ys):
                    if (((ind1-bestx)**2 + (ind2-besty)**2)**0.5 < search_radius):
                        possible_sources.append((ind1, ind2))
                print("Located %.0f possible sources." % len(possible_sources))
                if len(possible_sources) == 0:
                    # Skip the source selection step because no options were found. Give chance to update search parameters, too!
                    print("If your guess was definitely consistent with a source location,\ntry updating the threshold/search radius!")
                    check = int(input("Enter 1 to search again,\n2 to update the search parameters and then search again,\nor 3 to select manually: "))
                else:
                    # Ask the user to view one of the possible sources.
                    print("Please select the source from this list:")
                    for ind, item in enumerate(possible_sources):
                        print(str(ind) + "     " + str(item))
                    ind = int(input(""))
                    xs, ys = possible_sources[ind]

                    # Show the user where their chosen source is. Again, supercedes plot calls.
                    plt.subplot(1,1,1)
                    plt.imshow(d, vmin=0, vmax=100, origin='lower', cmap='binary_r')
                    plt.scatter(xs, ys, s=8, marker='x', color='red')
                    plt.xlim(int(xs-70), int(xs+70))
                    plt.ylim(int(ys-70), int(ys+70))
                    plt.title("Selected source")

                    plt.show(block=True)
                    plt.close()

                    # Ask the user to review the chosen source and decide if it is correct.
                    check = int(input("Enter 0 to keep this source,\n1 to search again,\n2 to update the search parameters and then search again,\nor 3 to select manually: "))
                if check == 0:
                    # We found the source!
                    satisfied = True
                if check == 1:
                    print("Restarting search with same search parameters...")
                if check == 2:
                    # Let the user update the search parameters and try again.
                    print("Current threshold: %.3f" % threshold)
                    threshold = int(input("Enter new threshold: "))
                    print("Current search radius: %.3f" % search_radius)
                    search_radius = int(input("Enter new search radius: "))
                    print("Current FWHM: %.3f" % fwhm)
                    fwhm = int(input("Enter new FWHM: "))
                if check == 3:
                    # Let the user manually set the source position.
                    xs = float(input("Enter x: "))
                    ys = float(input("Enter y: "))
                    plt.subplot(1,1,1)
                    plt.imshow(d, vmin=0, vmax=100, origin='lower', cmap='binary_r')
                    plt.scatter(xs, ys, s=8, marker="x", color="red")
                    plt.xlim(int(xs-70), int(xs+70))
                    plt.ylim(int(ys-70), int(ys+70))
                    plt.show(block=True)
                    plt.close()
                    # Ask them to review their selection.
                    check2 = int(input("Is this okay (0) or do you want to search again (1)?: "))
                    if check2 == 0:
                        satisfied = True
    print("Source selected:", xs, ys)
    return xs, ys
