from tqdm import tqdm
import numpy as np

def clean_spectra(specs, sigma):
    '''
    Compares all 1D spectra to a median spectra and replaces outliers with the median of that spectral point in time.
    
    :param specs: lst of np.array. The 1D spectra for each order as a function of time, with indices time and wavelength.
    :param sigma: float. Threshold of deviation from median spectral point, above which a pixel will be flagged as an outlier and masked.
    :return: oneD_spec array but with outliers masked.
    '''
    cleaned_specs = []
    # Iterate over spectra.
    for oneD_spec in tqdm(specs, desc='Cleaning spectral outliers... Progress:'):
        # Track outliers removed.
        bad_spex_removed = 0

        # Iteration stop condition. As long as outliers are being found, we have to keep iterating.
        outlier_found = True
        while outlier_found:
            # Define median spectrum in time and extend its size to include all time.
            med_spec = np.median(oneD_spec,axis=0)
            med_spec = np.array([med_spec,]*oneD_spec.shape[0])
            # Get standard deviation of each point.
            std_spec = np.std(oneD_spec,axis=0)
            std_spec = np.array([std_spec,]*oneD_spec.shape[0])

            # Flag outliers.
            S = np.where(np.abs(oneD_spec-med_spec) > sigma*std_spec, 1, 0)

            # Count outliers found.
            bad_spex_this_step = np.count_nonzero(S)
            bad_spex_removed += bad_spex_this_step

            if bad_spex_this_step == 0:
                # No more outliers found! We can break the loop now.
                outlier_found = False
            
            # Correct outliers and loop once more.
            oneD_spec = np.where(S == 1, med_spec, oneD_spec)
        print("1D spectral cleaning complete. Removed %.0f spectral outliers." % bad_spex_removed)
        cleaned_specs.append(oneD_spec)
    return cleaned_specs