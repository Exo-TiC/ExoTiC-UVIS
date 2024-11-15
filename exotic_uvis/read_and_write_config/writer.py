import os

def write_config(config_dict, run_name, stage, outdir):
    """Unpacks a dictionary and writes it out to a config file.

    Args:
        config_dict (dict): dictionary used to guide the execution of a
        Stage of ExoTiC-UVIS.
        run_name (str): name of this run, used to give the config file
        a unique and identifiable name.
        stage (int): which Stage was executed, which sets the template
        of the config file.
        outdir (outdir): path to where the config file is to be stored.
    """
    # Get correct print info.
    if stage == 0:
        header, subsection_headers, subsection_keys, subsection_comments = Stage0_info()
    if stage == 1:
        header, subsection_headers, subsection_keys, subsection_comments = Stage1_info()
    if stage == 2:
        header, subsection_headers, subsection_keys, subsection_comments = Stage2_info()
    if stage == 3:
        header, subsection_headers, subsection_keys, subsection_comments = Stage3_info()
    
    # And write.
    with open(os.path.join(outdir,"stage_{}_{}.hustle".format(stage, run_name)), mode='w') as f:
        print("Writing config file for Stage {}...".format(stage))
        # First, write the overall file header.
        f.write(header)
        f.write('\n\n')

        # Then, start parsing each step out (Setup, Step 1, Step 2, etc.).
        subsections = list(subsection_keys.keys())
        for i, subsection in enumerate(subsections):
            # Write the step name.
            f.write(subsection_headers[i])
            f.write('\n')

            # For every keyword and comment in that step...
            for j, keyword in enumerate(subsection_keys[subsection]):
                # Write the keyword, its stringed value, and the comment.
                try:
                    value = str(config_dict[keyword])
                    f.write("{0:<15} {1:<60} {2:}\n".format(keyword, value, subsection_comments[subsection][j]))
                except IndexError:
                    print(subsection, subsection_keys[subsection])
                    print("Index error here!")
            # A space between this step and the next step.
            f.write('\n')
        # Declare the file over.
        f.write("# ENDPARSE")
    print("Config file written.")
            

def Stage0_info():
    """Retrieves writer information for Stage 0.

    Returns:
        str, list, dict, dict: header and comments for writing the config.
    """

    header = "# ExoTiC-UVIS config file for launching Stage 0: Data Handling"

    subsection_headers = ["# Setup for Stage 0",
                          "# Step 1: Download files from MAST",
                          "# Step 2: Organizing files",
                          "# Step 3: Locating the target star",
                          "# Step 4: Quality quicklook",]
    
    subsection_keys = {"Setup":["toplevel_dir",
                                "verbose",
                                "show_plots",
                                "save_plots"],
                       "Step 1":["do_download",
                                 "programID",
                                 "target_name",
                                 "extensions"],
                       "Step 2":["do_organize",
                                 "visit_number",
                                 "filesfrom_dir"],
                       "Step 3":["do_locate",
                                 "location"],
                       "Step 4":["do_quicklook",
                                 "gif_dir"],
                       }
    
    subsection_comments = {"Setup":["# Directory where you want your files to be stored after Stage 0 has run. This is where /specimages, /directimages, /visitfiles, and /miscfiles will be stored.",
                                    "# Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.",
                                    "# Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.",
                                    "# Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.",],
                           "Step 1":["# Bool. Whether to perform this step.",
                                     '# ID of the observing program you want to query data from. On MAST, referred to as "proposal_ID".',
                                     '# Name of the target object you want to query data from. On MAST, referred to as "target_name".',
                                     "# lst of str or None. File extensions you want to download. If None, take all file extensions. Otherwise, take only the files specified. _flt.fits, _spt.fits recommended as minimum working case.",],
                           "Step 2":["# Bool. Whether to perform this step.",
                                     "# The visit number you want to operate on.",
                                     "# None or str. If you downloaded data in Step 1, leave this as None. If you have pre-downloaded data, please place all of it in filesfrom_dir. Don't sort it into sub-folders; ExoTiC-UVIS won't be able to find them if they are inside sub-folders!",],
                           "Step 3":["# Bool. Whether to perform this step.",
                                     "# None or tuple of float. Prior to running Stage 0, this will be None. After running Stage 0, a copy of this .hustle file will be made with this information included.",],
                           "Step 4":["# Bool. Whether to perform this step.",
                                     "# str. Where to save the quicklook gif to, relative to the toplevel_dir.",],
                           }
    return header, subsection_headers, subsection_keys, subsection_comments

def Stage1_info():
    """Retrieves writer information for Stage 1.

    Returns:
        str, list, dict, dict: header and comments for writing the config.
    """
    
    header = "# ExoTiC-UVIS config file for launching Stage 1: Reduction"

    subsection_headers = ["# Setup for Stage 1",
                          "# Step 1: Read in the data",
                          "# Step 2: Reject cosmic rays with time iteration\n# Step 2a: Fixed iteration parameters",
                          "# Step 2b: Free iteration parameters",
                          "# Step 2c: Sigma clip parameters",
                          "# Step 3: Reject hot pixels with spatial detection\n# Step 3a: Laplacian Edge Detection parameters",
                          "# Step 3b: Spatial smoothing parameters",
                          "# Step 4: Background subtraction\n# Step 4a: uniform value background subtraction",
                          "# Step 4b: Column-by-column background subtraction",
                          "# Step 4c: Pagul et al. background subtraction",
                          "# Step 5: Displacement estimation\n# Step 5a: Refine target location",
                          "# Step 5b: Source center-of-mass tracking",
                          "# Step 5c: Background star tracking",
                          "# Step 6: Quality quicklook",
                          "# Step 7: Save outputs",]
    
    subsection_keys = {"Setup":["toplevel_dir",
                                "run_name",
                                "verbose",
                                "show_plots",
                                "save_plots"],
                       "Step 1":["skip_first_fm",
                                 "skip_first_or",],
                       "Step 2a":["do_fixed_iter",
                                 "fixed_sigmas",
                                 "replacement",],
                       "Step 2b":["do_free_iter",
                                  "free_sigma"],
                       "Step 2c":["do_sigma_clip",],
                       "Step 3a":["do_led",
                                  "led_threshold",
                                  "led_factor",
                                  "led_n",
                                  "fine_structure",
                                  "contrast_factor",],
                       "Step 3b":["do_smooth",],
                       "Step 4a":["do_uniform",
                                  "fit",
                                  "bounds",
                                  "hist_min",
                                  "hist_max",
                                  "hist_bins"],
                       "Step 4b":["do_column",
                                  "rows",
                                  "mask_trace",
                                  "dist_from_trace",
                                  "col_sigma",],
                       "Step 4c":["do_Pagul",
                                  "path_to_Pagul",
                                  "mask_parameter",
                                  "smooth_fits",
                                  "smoothing_param",
                                  "median_columns",],
                       "Step 5a":["do_location",],
                       "Step 5b":["do_0thtracking",
                                  "location",],
                       "Step 5c":["do_bkg_stars",
                                  "bkg_stars_loc",],
                       "Step 6":["do_quicklook",
                                 "gif_dir",],
                       "Step 7":["do_save",],
                       }
    
    subsection_comments = {"Setup":["# Directory where your Stage 0 files are stored. This folder should contain the specimages/, directimages/, etc. folders with your data.",
                                    "# Str. This is the name of the current run. It can be anything that does not contain spaces or special characters (e.g. $, %, @, etc.).",
                                    "# Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.",
                                    "# Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.",
                                    "# Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.",],
                           "Step 1":["# Bool. If True, ignores all first frames in each orbit.",
                                     "# Bool. If True, ignores all frames in the first orbit.",],
                           "Step 2a":["# Bool. Whether to use fixed iteration rejection to clean the timeseries.",
                                      "# lst of float. The sigma to reject outliers at in each iteration. The length of the list is the number of iterations.",
                                      "# int or None. If int, replaces flagged outliers with the median of values within +/-replacement indices of the outlier. If None, uses the median of the whole timeseries instead.",],
                           "Step 2b":["# Bool. Whether to use free iteration rejection to clean the timeseries.",
                                      "# float. The sigma to reject outliers at in each iteration. Iterates over each pixel's timeseries until no outliers at this sigma level are found.",],
                           "Step 2c":["# Bool. Whether to use sigma clipping rejection to clean the timeseries.",],
                           "Step 3a":["# Bool. Whether to use Laplacian Edge Detection rejection to clean the frames.",
                                      "# Float. The threshold parameter at which to kick outliers in LED. The lower the number, the more values will be replaced.",
                                      "# Int. The subsampling factor. Minimum value 2. Higher values increase computation time but aren't expected to yield much improvement in rejection.",
                                      "# Int. Number of times to do LED on each frame. Enter None to continue performing LED on each frame until no outliers are found.",
                                      "# Bool. Whether to build a fine structure model, which can protect narrow bright features like traces from LED.",
                                      "# Float. If fine_structure is True, acts as the led_threshold for the fine structure step.",],
                           "Step 3b":["# Bool. Whether to use spatial smoothing rejection to clean the frames.",],
                           "Step 4a":["# Bool. Whether to subtract the background using one uniform value as the value for the entire frame.",
                                      "# Str. The value to extract from the histogram. Options are None (to extract the mode), 'Gaussian' (to fit the mode with a Gaussian), or 'median' (to take the median within hist_min < v < hist_max).",
                                      "# Lst of lst of float. The region from which the background values will be extracted. Each list consists of [x1,x2,y1,y2]. If None, simply uses the full frame.",
                                      "# Float. Minimum value to consider for the background. Leave as None to use min(data).",
                                      "# Float. Maximum value to consider for the background. Leave as None to use max(data).",
                                      "# Int. Number of histogram bins for background subtraction.",],
                           "Step 4b":["# Bool. Whether to subtract the background using a column-by-column method.",
                                      "# list of int. The indices defining the rows used as background.",
                                      "# Bool. If True, ignores rows parameter and instead masks the traces and 0th order to build a background region.",
                                      "# Int. If mask_trace is True, this is how many rows away a pixel must be from the trace to qualify as background.",
                                      "# float. How aggressively to mask outliers in the background region.",],
                           "Step 4c":["# Bool. Whether to subtract the background using the scaled Pagul et al. G280 sky image.",
                                      "# Str. The absolute path to where the Pagul et al. G280 sky image is stored.",
                                      "# Float. How strong the trace masking should be. Smaller values mask more of the image.",
                                      '# Bool. If True, smooths the values of the Pagul et al. fit parameter in time. Helps prevent background "flickering".',
                                      '# Float. Sigma for smoothing the fit parameter. Smaller sigma means more smoothing.',
                                      "# Bool. If True, takes the median value of each column in the Pagul et al. sky image as the background. As the Pagul et al. 2023 image is undersampled, this helps to suppress fluctuations in the image.",],
                           "Step 5a":["# Bool. Whether the location of the target in the direct image extracted from Stage 0 should be refined by fitting.",],
                           "Step 5b":["# Bool. Whether to track frame displacements by centroiding the 0th order.",
                                      "# lst of float. Initial guess for the location of the target star. You can use this to bypass location fitting in Stage 1.",],
                           "Step 5c":["# Bool. Whether to track frame displacements by centroiding background stars.",
                                      "# Lst of lst of float. Every list should indicate the estimated location of every background star",],
                           "Step 6":["# Bool. Whether to perform this step.",
                                     "# str. Where to save the quicklook gif to."],
                           "Step 7":["# Bool. If True, saves the output xarray to be used in Stage 2.",],
                           }
    return header, subsection_headers, subsection_keys, subsection_comments

def Stage2_info():
    """Retrieves writer information for Stage 2.

    Returns:
        str, list, dict, dict: header and comments for writing the config.
    """

    header = "# ExoTiC-UVIS config file for launching Stage 2: Extraction"

    subsection_headers = ["# Setup for Stage 2",
                          "# Step 1: Read in the data",
                          "# Step 2: Trace configuration",
                          "# Step 3: 1D spectral extraction",
                          "# Step 3a: Box extraction parameters",
                          "# Step 3b: Optimum extraction parameters",
                          "# Step 4: 1D spectral cleaning and aligning",
                          ]
    
    subsection_keys = {"Setup":["toplevel_dir",
                                "run_name",
                                "verbose",
                                "show_plots",
                                "save_plots"],
                       "Step 1":[],
                       "Step 2":["path_to_cal",
                                 "traces_to_conf",
                                 "refine_fit"],
                       "Step 3":["method",
                                  "subtract_contam",
                                  "sens_correction",
                                  "mask_objs",],
                       "Step 3a":["determine_hw",
                                  "indices",
                                  "halfwidths_box",],
                       "Step 3b":["aperture_type",
                                  "halfwidths_opt",],
                       "Step 4":["outlier_sigma",
                                 "align",],
                       }
    
    subsection_comments = {"Setup":["# Directory where your current project files are stored. This folder should contain the specimages/, directimages/, etc. folders with your data as well as the outputs folder.",
                                    "# Str. This is the name of the current run. It can be anything that does not contain spaces or special characters (e.g. $, %, @, etc.).",
                                    "# Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.",
                                    "# Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.",
                                    "# Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.",],
                           "Step 1":[],
                           "Step 2":["# Str. The absolute path to the .conf file used by GRISMCONF for the chip your data were taken on.",
                                     "# Lst of str. The traces you want to configure and extraction from.",
                                     "# Bool. If True, uses Gaussian fitting to refine the trace solution.",],
                           "Step 3":["# Str. Options are 'box' (draw a box around the trace and sum without weights) or 'optimum' (weight using Horne 1986 methods).",
                                     "# Bool. Whether to model the contaminating orders and subtract them from your trace during extraction. Sometimes works, sometimes just adds lots of scatter.",
                                     "# Bool. Whether to correct for the G280's changing sensitivity as a function of wavelength. Since absolute calibrated spectra aren't needed in exoplanetary sciences, you can skip this safely.",
                                     "# List of lists. If there are background objects in your planned aperture, mask them here. Each entry is (x,y,radius).",],
                           "Step 3a":["# Bool. If True, automatically determines preferred half-width for each order by minimizing out-of-transit/eclipse residuals.",
                                      "# Lst of lsts of int. If determine_hw, these are the indices used to estimate the out-of-transit/eclipse residuals.",
                                      "# Lst of ints. The half-width of extraction aperture to use for each order. Input here is ignored if 'determine_hw' is True.",],
                           "Step 3b":["# Str. Type of aperture to draw. Options are 'row_polyfit', 'column_polyfit', 'column_gaussfit', 'column_moffatfit', 'median', 'smooth'.",
                                      "# Lst of ints. The half-width of extraction aperture to use for each order. For optimum extraction, you should make this big (>12 pixels at least). There is no 'preferred' half-width in optimum extraction due to the weights.",],
                           "Step 4":["# Float. Sigma at which to reject spectral outliers in time. Outliers are replaced with median of timeseries. Enter False to skip this step.",
                                     "# Bool. If True, uses cross-correlation to align spectra to keep wavelength solution consistent.",],
                           }
    return header, subsection_headers, subsection_keys, subsection_comments

def Stage3_info():
    """Retrieves writer information for Stage 3.

    Returns:
        str, list, dict, dict: header and comments for writing the config.
    """

    header = "# ExoTiC-UVIS config file for launching Stage 3: Binning"

    subsection_headers = ["# Setup for Stage 3",
                          "# Step 1: Read in the data",
                          "# Step 2: Light curve extraction",
                          ]
    
    subsection_keys = {"Setup":["toplevel_dir",
                                "run_name",
                                "verbose",
                                "show_plots",
                                "save_plots"],
                       "Step 1":["orders",],
                       "Step 2":["bin_method",
                                 "wavelength_bins",
                                 "N_columns",
                                 "reject_bad_cols",
                                 "bad_col_thres"],
                       "Step 3":["time_binning",
                                 "sigma_clip",
                                 "normalize",],
                       }
    
    subsection_comments = {"Setup":["# Directory where your current project files are stored. This folder should contain the specimages/, directimages/, etc. folders with your data as well as the outputs folder.",
                                    "# Str. This is the name of the current run. It can be anything that does not contain spaces or special characters (e.g. $, %, @, etc.).",
                                    "# Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.",
                                    "# Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.",
                                    "# Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.",],
                           "Step 1":["# List of string. The orders you want to load and operate on.",],
                           "Step 2":["# Str. How to bin the light curves. Options are 'columns' (bin N columns at a time) or 'wavelengths' (bin from wavelength1 to wavelength2).",
                                     "# Lst of floats or numpy array. If bin_method is 'wavelengths', defines edges of each wavelength bin.",
                                     "# Int. If bin_method is 'columns', how many columns go into each bin.",
                                     "# bool. If True, masks contributions from columns deemed too noisy.",
                                     "# float. Used to control how aggressively we flag columns. The lower the number, the less noisiness we tolerate in our columns.",],
                           "Step 3":["# Int or None. If int, how many frames in time should be binned. Reduces computation time but degrades time resolution.",
                                     "# Float or None. If float, the sigma at which to mask outliers in sigma clipping.",
                                     "# Bool. If True, normalizes curves by out-of-transit/eclipse flux.",],
                           }
    return header, subsection_headers, subsection_keys, subsection_comments
