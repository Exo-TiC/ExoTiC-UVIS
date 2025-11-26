Stage 1: Reduction
======================

**NOTE: Make sure you ran Stage 0: Data Handling before attempting this stage!**
  
With the data in hand, it is time to clean the data of cosmic rays, hot pixels, and background signal. :code:`HUSTLE-tools` Stage 1 takes the files downloaded in Stage 1 and attempts to correct sources of noise to improve the signal-to-noise ratio of your extracted spectra. This tutorial will walk you through this process.

Creating the Stage 1 configuration file
---------------------------------------

The first step is to create the configuration file that will guide the execution of Stage 1. Create a folder to store the configuration file in, e.g. :code:`configs/`. Then create :code:`configs/stage_1_input_config.hustle` and populate it with the following:

.. code-block:: bash

  # HUSTLE-tools config file for launching Stage 1: Reduction
  
  # Setup for Stage 1
  toplevel_dir    'output'                                    # Directory where your Stage 0 files are stored. This folder should contain the specimages/, directimages/, etc. folders with your data.
  output_run      'run1'                                      # Str. This is the name to save the current run to. It can be anything that does not contain spaces or special characters (e.g. $, %, @, etc.).
  verbose         2                                           # Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.
  show_plots      2                                           # Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.
  save_plots      2                                           # Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.
  
  # Step 1: Read in the data
  skip_first_fm   False                                       # Bool. If True, ignores all first frames in each orbit.
  skip_first_or   False                                       # Bool. If True, ignores all frames in the first orbit.
  
  # Step 2: Reject cosmic rays with time iteration
  # Step 2a: Fixed iteration parameters
  do_fixed_iter   True                                        # Bool. Whether to use fixed iteration rejection to clean the timeseries.
  fixed_sigmas    [10,10]                                     # lst of float. The sigma to reject outliers at in each iteration. The length of the list is the number of iterations.
  replacement     7                                           # int or None. If int, replaces flagged outliers with the median of values within +/-replacement indices of the outlier. If None, uses the median of the whole timeseries instead.
  
  # Step 2b: Free iteration parameters
  do_free_iter    False                                       # Bool. Whether to use free iteration rejection to clean the timeseries.
  free_sigma      3.5                                         # float. The sigma to reject outliers at in each iteration. Iterates over each pixel's timeseries until no outliers at this sigma level are found.
  
  # Step 3: Reject hot pixels with spatial detection
  # Step 3a: Laplacian Edge Detection parameters
  do_led          True                                        # Bool. Whether to use Laplacian Edge Detection rejection to clean the frames.
  led_threshold   5                                           # Float. The threshold parameter at which to kick outliers in LED. The lower the number, the more values will be replaced.
  led_factor      2                                           # Int. The subsampling factor. Minimum value 2. Higher values increase computation time but aren't expected to yield much improvement in rejection.
  led_n           2                                           # Int. Number of times to do LED on each frame. Enter None to continue performing LED on each frame until no outliers are found.
  fine_structure  True                                        # Bool. Whether to build a fine structure model, which can protect narrow bright features like traces from LED.
  contrast_factor 5                                           # Float. If fine_structure is True, acts as the led_threshold for the fine structure step.
  
  # Step 3b: Spatial smoothing parameters
  do_smooth       False                                       # Bool. Whether to use spatial smoothing rejection to clean the frames.
  smth_type       '1D_smooth'                                 # Str. Type of spatial correction to be applied. Options are  '1D_smooth', '2D_smooth', and 'polyfit'.
  smth_kernel     11                                          # Int or tuple. The kernel to use for building the median-filtered image. If using 1D_smooth, should be an odd int. If using 2D_smooth, should be a tuple of two odd ints.
  smth_threshold  5                                           # Float. If an image pixel deviates from the median-filtered image by this threshold, kick it from the image. The lower the value, the more pixels get kicked.
  smth_bounds     [[260, 370, 640, 1100],]                    # Lst of lst of float. The regions that will be corrected for bad pixels. Each list consists of [x1,x2,y1,y2]. If None, simply corrects the full frame.
  
  # Step 4: Background subtraction
  # Step 4a: uniform value background subtraction
  do_uniform      True                                        # Bool. Whether to subtract the background using one uniform value as the value for the entire frame.
  fit             'Gaussian'                                  # Str. The value to extract from the histogram. Options are None (to extract the mode), 'Gaussian' (to fit the mode with a Gaussian), or 'median' (to take the median within hist_min < v < hist_max).
  bounds          [[0,150,0,400],[440,590,0,400]]             # Lst of lst of float. The region from which the background values will be extracted. Each list consists of [x1,x2,y1,y2]. If None, simply uses the full frame.
  hist_min        -20                                         # Float. Minimum value to consider for the background. Leave as None to use min(data).
  hist_max        50                                          # Float. Maximum value to consider for the background. Leave as None to use max(data).
  hist_bins       1000                                        # Int. Number of histogram bins for background subtraction.
  
  # Step 4b: Column-by-column background subtraction
  do_column       False                                       # Bool. Whether to subtract the background using a column-by-column method.
  rows            [i for i in range(10)]                      # list of int. The indices defining the rows used as background.
  mask_trace      True                                        # Bool. If True, ignores rows parameter and instead masks the traces and 0th order to build a background region.
  dist_from_trace 100                                         # Int. If mask_trace is True, this is how many rows away a pixel must be from the trace to qualify as background.
  col_sigma       3                                           # float. How aggressively to mask outliers in the background region.
  
  # Step 4c: Pagul et al. background subtraction
  do_Pagul        False                                       # Bool. Whether to subtract the background using the scaled Pagul et al. G280 sky image.
  path_to_Pagul   './'                                        # Str. The absolute path to where the Pagul et al. G280 sky image is stored.
  mask_parameter  0.001                                       # Float. How strong the trace masking should be. Smaller values mask more of the image.
  smooth_fits     True                                        # Bool. If True, smooths the values of the Pagul et al. fit parameter in time. Helps prevent background "flickering".
  smoothing_param 2.5                                         # Float. Sigma for smoothing the fit parameter. Smaller sigma means more smoothing.
  median_columns  True                                        # Bool. If True, takes the median value of each column in the Pagul et al. sky image as the background. As the Pagul et al. 2023 image is undersampled, this helps to suppress fluctuations in the image.
  
  # Step 5: Displacement estimation
  # Step 5a: Refine target location
  do_location     True                                        # Bool. Whether the location of the target in the direct image extracted from Stage 0 should be refined by fitting.
  
  # Step 5b: Source center-of-mass tracking
  do_0thtracking  True                                        # Bool. Whether to track frame displacements by centroiding the 0th order.           
  location        [970, 170]                                  # lst of float. Initial guess for the location of the target star. You can use this to bypass location fitting in Stage 1.
  
  # Step 5c: Background star tracking
  do_bkg_stars    False                                       # Bool. Whether to track frame displacements by centroiding background stars.
  bkg_stars_loc   [[0, 0], [0, 0]]                            # Lst of lst of float. Every list should indicate the estimated location of every background star.
  
  # Step 6: Quality quicklook
  do_quicklook    True                                        # Bool. Whether to perform this step.
  
  # Step 7: Save outputs
  do_save         True                                        # Bool. If True, saves the output xarray to be used in Stage 2.
  
  # ENDPARSE

Let's break down each of these steps to make sure we understand what we can tune in this stage.

Setup for Stage 1
~~~~~~~~~~~~~~~~~
You can customize the :code:`toplevel_dir` folder to be any name you like as long as you use the same folder name for all stages. You will likely run multiple reductions of the data as you explore different cleaning strategies. To keep each run separate so that no files get overwritten, you can adjust the :code:`output_run` variable each time you run Stage 1. This will cause the outputs to go to a subfolder with the name :code:`toplevel_dir/outputs/stage_1/output_run`. The :code:`verbose`, :code:`show_plots`, and :code:`save_plots` keys respectively control the level of detail in the statements printed by the pipeline while it executes, the number of plots opened in the terminal or .ipynb notebook cell you are running the script in, and the number of plots saved out to .png or .gif files to be reviewed at any time after execution.

Step 1: Read in the data
~~~~~~~~~~~~~~~~~~~~~~~~
HST WFC3 data is known to present strong systematics that may vary between orbits. For WFC3/IR data, strong systematics may require that the entire first orbit be removed from analysis to maximize the quality of the retrieved spectra. This can be done by setting :code:`skip_first_or` to True. For HST WFC3/UVIS, most observations do not suffer from systematics that affect the entire first orbit, and so you can usually get away with setting :code:`skip_first_or` to False. However, the first frame of each orbit has been seen in some observations to possess enhanced systematics, requiring that the first frames be discarded. If your data suffer from enhanced first-frame systematics, you can remove them from analysis by setting :code:`skip_first_fm` to True.

Step 2: Reject cosmic rays with time iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cosmic rays, or energetic particles striking the detector at random and originating from unknown astrophysical sources, contribute additional noise to extracted spectra. :code:`HUSTLE-tools` offers two routines for treating cosmic rays in your data: fixed iteration rejection and free iteration rejection.

Step 2a: Fixed iteration parameters
'''''''''''''''''''''''''''''''''''
Fixed iteration rejection iterates over every pixel's time series a fixed number of times, using a specified threshold at each iteration to reject outliers. Set :code:`do_fixed_iter` to True if you want to use this method. The number of iterations, and the threshold for rejection at each iteration, is specified by supplying a list of thresholds to the :code:`fixed_sigmas` variable. When outliers are found, their handling is controlled by the :code:`replacement` variable, which can be set to an integer to specify the half-width of the running median window used to compute the replacement value, or can be set to None to replace the outlier by the overall median of the pixel's time series.

Step 2b: Free iteration parameters
''''''''''''''''''''''''''''''''''
Free iteration rejection iterates over every pixel's time series at a single rejection threshold, and will continue iterating at that threshold until no outliers are found. Set :code:`do_free_iter` to True if you want to use this method. The threshold that will be used on these iterations is set as a float supplied to the :code:`free_sigma` variable.
  

Step 3: Locating the target star
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set :code:`do_locate` to True to prompt the pipeline to present the direct image so that you can locate the target star within this frame. This is the only step of the pipeline which requires user input, but it is necessary to ensure successful wavelength calibration in Stage 2. If you have already located the coordinates of the target star, you can set :code:`do_locate` to False and simply supply the coordinates as a tuple of floats to the :code:`location` variable. Otherwise, a copy of this configuration file will be output after execution which will update the :code:`location` variable to the coordinates selected during the :code:`do_locate` step.

Step 4: Quality quicklook
~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to save a "quicklook" gif that presents all of the data frames in succession as well as simple diagnostics of the image and trace flux, simply set :code:`do_quicklook` to True.

Running Stage 0
---------------

With the configuration file created and stored in :code:`configs/stage_0_input_config.hustle`, create a simple .py or .ipynb script with the following contents:

.. code-block:: bash

  from hustle_tools import run_pipeline
  
  config_files_dir = "configs"
  stages = (0,)
  
  run_pipeline(config_files_dir, stages)

Then execute this script to run Stage 0! The output in your cell should look similar to the output shown below, where we have used HST-GO 15288 (PI: David Sing), visit 01, target HAT-P-41B as an example:

.. code-block:: bash

  '''will come back to this later :3'''

Assessing Stage 0's success
---------------------------
Stage 0 is the simplest stage that has very few diagnostics to look over. You will know if Stage 0 succeeded if:

  1. The :code:`toplevel_dir` folder has been created and populated with the :code:`specimages`, :code:`directimages`, :code:`visitfiles`, :code:`miscfiles`, and :code:`outputs` subfolders.
  2. The :code:`toplevel_dir/outputs/stage_0` folder contains an updated copy of the .hustle configuration folder with the :code:`location` variable changed from None to a tuple of floats.
  3. The quicklookup.gif created by this stage, or the .fits files downloaded to the :code:`toplevel_dir/specimages` directory, clearly show your target star and contain all of the orbits and total number of frames you expected.
