import numpy as np
from scipy.ndimage import median_filter

def spatial_smoothing(obs, sigma=10):
    '''
    For Carlos.
    '''
    return obs

def laplacian_edge_detection(obs, sigma=10, factor=2, n=2, build_fine_structure=False, contrast_factor=5,
                             verbose = 0, show_plots = 0, save_plots = 0, output_dir = None):
    '''
    Convolves a Laplacian kernel with the obs.images to replace spatial outliers with
    the median of the surrounding 3x3 kernel.
    
    :param obs: xarray. Its obs.images DataSet contains the images, and its obs.errors DataSet is used to estimate readnoise. The obs.data_quality DataSet will be updated where outliers are detected.
    :param sigma: float. Threshold of deviation from median of Laplacian image, above which a pixel will be flagged as an outlier and masked.
    :param factor: int. Factor by which to resample the array. Must be at least 2.
    :param n: int. Times to iterate over the data. If None, iterate until no new outliers are flagged.
    :param build_fine_structure: bool. If True, builds a fine structure model which protects data that varies on small lengthscales from being attacked by LED.
    :param contrast_factor: float. If build_fine_structure is True, this is the threshold of deviation we need to exceed in L+/F to flag outliers.
    :return: obs xarray with outliers removed and data quality flags updated.
    '''
    # Define the Laplacian kernel.
    l = 0.25*np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

    # Iterate over each frame one at a time until the iteration stop condition is met by each frame.
    print("Cleaning threshold=%.1f outliers with Laplacian edge detection..." % sigma)
    for k in range(obs.images.shape[0]):
        # Get the frame, errors, and dq array as np.array objects so we can operate on them.
        data_frame = obs.images[k].values
        errs = obs.errors[k].values
        dq = obs.data_quality[k].values

        # Track outliers flagged in this frame and iterations performed.
        bad_pix_removed = 0
        iteration_N = 1

        # Then start iterating over this frame and keep going until the iteration stop condition is met.
        stop_iterating = False
        while not stop_iterating:
            # Estimate readnoise value.
            var2 = errs**2 - data_frame
            var2[var2 < 0] = 0 # enforce positivity.
            rn = np.sqrt(var2) # estimate readnoise array

            # Build the noise model.
            noise_model = build_noise_model(data_frame, rn)
            if build_fine_structure:
                F = build_fine_structure_model(data_frame)

            # Subsample the array.
            subsample, original_shape = subsample_frame(data_frame, factor=factor)
            
            # Convolve subsample with laplacian.
            lap_img = np.convolve(l.flatten(),subsample.flatten(),mode='same').reshape(subsample.shape)
            lap_img[lap_img < 0] = 0 # force positivity
            
            # Resample laplacian-convolved subsampled image to original size.
            resample = resample_frame(lap_img, original_shape)

            # Divide by the noise model scaled by the resampling factor.
            S = resample/(factor*noise_model)
            
            # Remove sampling flux to protect data from being targeted by LED.
            S = S - median_filter(S, size=5)

            # Spot outliers.
            S[np.abs(S) < sigma] = 0 # any not zero after this are rays.
            S[S!=0] = 1 # for visualization and comparison to fine structure model.

            # If we have a fine structure model, we also need to check the contrast.
            if build_fine_structure:
                contrast_image = resample/F
                contrast_image[contrast_image < contrast_factor] = 0 # any not zero after this are rays.
                contrast_image[contrast_image!=0] = 1 # for visualization and comparison to sampling flux model.
                
                # Then we need to merge the results of S = Laplacian_image/factor*noise_model - sampling_flux
                # and contrast_image = Laplacian_image/Fine_structure_model so that we only take where both are 1.
                S = np.where(S == contrast_image, S, 0)

            # Ignore the 0th order, it's a dead end of endless masking.
            xmid = int(S.shape[1]/2)
            S[0:-1,xmid-70:xmid+70] = 0 # FIX: currently hardcoded to assume the source / 0th order is near the middle of the frame.

            # Report where data quality flags should be added and count pixels to be replaced.
            dq = np.where(S != 0, 1, dq)
            bad_pix_last_frame = -100
            if iteration_N != 1:
                bad_pix_last_frame = bad_pix_this_frame
            bad_pix_this_frame = np.count_nonzero(S)
            bad_pix_removed += bad_pix_this_frame

            # Report progress.
            print("Bad pixels removed on iteration %.0f: %.0f" % (iteration_N, bad_pix_this_frame))

            # Correct frames.
            med_filter_image = median_filter(data_frame,size=5)
            data_frame = np.where(S != 0, med_filter_image, data_frame)

            # Increment iteration number and check if condition to stop iterating is hit.
            iteration_N += 1
            if (n != None and iteration_N > n): # if it has hit the iteration cap
                stop_iterating = True
            if (n == None and bad_pix_this_frame == bad_pix_last_frame): # if it has stalled out on finding new outliers
                stop_iterating = True
            
        print("Finished cleaning frame %.0f in %.0f iterations." % (k, iteration_N-1))
        print("Total pixels corrected: %.0f out of %.0f" % (bad_pix_removed, S.shape[0]*S.shape[1]))
        # Now replace the xarray datasets with the corrected frame and updated dq array.
        obs.images[k] = obs.images[k].where(obs.images[k].values == data_frame,data_frame)
        obs.data_quality[k] = obs.data_quality[k].where(obs.data_quality[k].values == dq,dq)
    print("All frames cleaned of spatial outliers by LED.")
    return obs

def build_noise_model(data_frame, readnoise):
    '''
    Builds a noise model for the given data frame, following van Dokkum 2001 methods.

    :param data_frame: 2D array. Frame from the images DataSet, used to build the noise model.
    :param readnoise: float. Readnoise estimated to be in the data frame.
    :return: 2D array same size as the data frame, a noise model describing noise in the frame.
    '''
    noise_model = np.sqrt(median_filter(np.abs(data_frame),size=5)+readnoise**2)
    noise_model[noise_model <= 0] = np.mean(noise_model) # really want to avoid nans
    return noise_model

def subsample_frame(data_frame, factor=2):
    '''
    Subsamples the input frame by the given subsampling factor.

    :param data_frame: 2D array. Frame from the DN array.
    :param factor: int >= 2. Factor by which to subsample the array.
    :return: 2D array same shape as data frame, subsampled by factor.
    '''
    if factor < 2:
        print("Subsampling factor must be at least 2, forcing factor to 2...")
        factor = 2 # Force factor 2 or more
    factor = int(factor) # Force integer
    
    original_shape = np.shape(data_frame)
    ss_shape = (original_shape[0]*factor,original_shape[1]*factor)
    subsample = np.empty(ss_shape)
    
    # Subsample the array.
    for i in range(ss_shape[0]):
        for j in range(ss_shape[1]):
            try:
                subsample[i,j] = data_frame[int((i+1)/2),int((j+1)/2)]
            except IndexError:
                subsample[i,j] = 0
    return subsample, original_shape

def resample_frame(data_frame, original_shape):
    '''
    Resamples a subsampled array back to the original shape.

    :param data_frame: 2D array. Subsampled frame from the images DataSet.
    :param original_shape: tuple of int. Original shape of the subsampled array.
    :return: 2D array with original shape resampled from the data frame.
    '''
    resample = np.empty(original_shape)
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            resample[i,j] = 0.25*(data_frame[2*i-1,2*j-1] +
                                  data_frame[2*i-1,2*j]   +
                                  data_frame[2*i,2*j-1]   +
                                  data_frame[2*i,2*j])
    return resample

def build_fine_structure_model(data_frame):
    '''
    Builds a fine structure model for the data frame.

    :param data_frame: 2D array. Native resolution data.
    :return: 2D array of fine structure model.
    '''
    F = median_filter(data_frame, size=3) - median_filter(median_filter(data_frame, size=3), size=7)
    F[F <= 0] = np.mean(F) # really want to avoid nans
    return F