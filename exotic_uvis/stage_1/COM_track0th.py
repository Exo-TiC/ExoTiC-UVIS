from photutils.centroids import centroid_com

def track0th(obs, guess):
    '''
    Tracks the 0th order through all frames using centroiding.
    
    :param obs: xarray. Its obs.images DataSet contains the images.
    :param guess: lst of float. Initial x, y position guess for the 0th order's location.
    :return: location of the direct image in x, y floats.
    '''    
    # Correct direct image guess to be appropriate for spec images.
    # FIX: hardcoded based on WASP-31 test. There must be a better way...
    guess[0] += 100
    guess[1] += 150

    # Open lists of position.
    X, Y = [], []
    for k in range(obs.images.shape[0]):
        # Open the kth image.
        d = obs.images[k].values

        # Unpack guess and integerize it.
        x0, y0 = [int(i) for i in guess]

        # Clip a window near the guess.
        window = d[y0-70:y0+70,x0-70:x0+70]

        # Centroid the window.
        xs, ys = centroid_com(window)
        print(xs, ys)

        # Return to native window.
        xs += x0 - 70
        ys += y0 - 70
        
        # Take the source.
        X.append(xs)
        Y.append(ys)
    print("Tracked 0th order in %.0f frames." % obs.images.shape[0])
    return X, Y