import numpy as np
from tqdm import tqdm

def fixed_iteration_rejection(obs, sigmas=[10,10], replacement=None):
    '''
    Iterates a fixed number of times using a different sigma at each iteration to reject cosmic rays.

    :param obs: xarray. Its obs.images DataSet contains the images.
    :param sigmas: lst of int. Sigma to use for each iteration. len(sigmas) is the number of iterations that will be run.
    :param_replacement: int or None. If None, replace outlier pixels with median in time. If int, replace with median of int values either side in time.
    :return: obs with cosmic rays removed.
    '''
    # Track pixels corrected.
    bad_pix_removed = 0
    # Iterate over each sigma.
    for j, sigma in enumerate(sigmas):
        # Get the median time frame and std as a reference.
        d_all = obs.images[:].values
        med = np.median(d_all,axis=0)
        std = np.std(d_all,axis=0)

        # Track outliers flagged by this sigma.
        bad_pix_this_sigma = 0

        # Then check over frames and see where outliers are.
        for k in tqdm(range(obs.images.shape[0]), desc = "Correcting for %.0fth sigma... Progress:" % j):
            # Get the frame and dq array as np.array objects so we can operate on them.
            d = obs.images[k].values
            dq = obs.data_quality[k].values
            S = np.where(np.abs(d - med) > sigma*std, 1, 0)
            
            # Report where data quality flags should be added and count pixels to be replaced.
            dq = np.where(S != 0, 1, dq)
            bad_pix_this_frame = np.count_nonzero(S)
            bad_pix_this_sigma += bad_pix_this_frame

            # If replacement is not None, custom replacement.
            correction = med
            if replacement:
                # Take the median of the frames that are +/- replacement away from the current frame.
                l = k - replacement
                r = k + replacement
                # Cut at edges.
                if l < 0:
                    l = 0
                if r > obs.images.shape[0]:
                    r = obs.images.shape[0]
                correction = np.median(d_all[l:r,:,:],axis=0)
            # Correct frame and replace obs.images frame with the new array.
            d = np.where(S == 1, correction, d)
            obs.images[k] = obs.images[k].where(obs.images[k].values == d,d)
            obs.data_quality[k] = obs.data_quality[k].where(obs.data_quality[k].values == dq,dq)
        
        print("Bad pixels removed on iteration %.0f with sigma %.2f: %.0f" % (j, sigma, bad_pix_this_sigma))
        bad_pix_removed += bad_pix_this_sigma
    print("All iterations complete. Total pixels corrected: %.0f out of %.0f" % (bad_pix_removed, S.shape[0]*S.shape[1]))
    return obs