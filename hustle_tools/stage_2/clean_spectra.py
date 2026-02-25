import numpy as np
import matplotlib.pyplot as plt

#define plotting parameters
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
plt.rc('legend',**{'fontsize':11})


def clean_spectra(spec, sigma,
                  verbose = 0):
    """Replace spectral outliers in time with their temporal median.

    Args:
        spec (array-like): spectrum to clean.
        sigma (float): threshold at which to reject a value as an outlier.
        verbose (int, optional): How detailed you want the printed statements to be. Defaults to 0.

    Returns:
        array-like: input spectrum cleaned of final outliers.
    """

    if verbose > 0:
        print("Cleaning spectral outliers...")
    
    # Array-ify input spec for use of numpy.
    oneD_spec = np.array(spec)

    # Track outliers removed.
    bad_spex_removed = 0
    n_spex_points = oneD_spec.shape[0]*oneD_spec.shape[1]

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

    if verbose > 0:
        print("1D spectral cleaning complete. Removed %.0f spectral outliers from %.0f spectral points." % (bad_spex_removed,
                                                                                                            n_spex_points))

    return oneD_spec
