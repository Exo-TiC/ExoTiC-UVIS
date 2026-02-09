Stage 1: Reduction
======================
Prefer Jupyter notebooks? A .ipynb of this tutorial is also available on `GitHub <https://github.com/Exo-TiC/HUSTLE-tools/blob/main/tutorials/HUSTLE_tools_stage1.ipynb>`_!

**NOTE: Make sure you ran Stage 0: Data Handling before attempting this stage!**
  
With the data in hand, it is time to clean the data of cosmic rays, hot pixels, and background signal. :code:`HUSTLE-tools` Stage 1 takes the files downloaded in Stage 1 and attempts to correct sources of noise to improve the signal-to-noise ratio of your extracted spectra. This tutorial will walk you through this process.

Creating the Stage 1 configuration file
---------------------------------------

The first step is to create the configuration file that will guide the execution of Stage 1. Create a folder to store the configuration file in, e.g. :code:`configs/`. Then create :code:`configs/stage_1_input_config.hustle` and populate it with the following template:

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
Fixed iteration rejection iterates over every pixel's time series a fixed number of times, using a specified threshold at each iteration to reject outliers. Set :code:`do_fixed_iter` to True if you want to use this method. The number of iterations, and the threshold for rejection at each iteration, is specified by supplying a list of thresholds to the :code:`fixed_sigmas` variable. When outliers are found, their handling is controlled by the :code:`replacement` variable, which can be set to an integer to specify the half-width of the running median window used to compute the replacement value, or can be set to None to replace the outlier by the median of the pixel's full time series.

Step 2b: Free iteration parameters
''''''''''''''''''''''''''''''''''
Free iteration rejection iterates over every pixel's time series at a single rejection threshold, and will continue iterating at that threshold until no outliers are found. Set :code:`do_free_iter` to True if you want to use this method. The threshold that will be used on these iterations is set as a float supplied to the :code:`free_sigma` variable.

Step 3: Reject hot pixels with spatial detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hot pixels output too much signal compared to their neighbors, while cold/dead pixels output little to no signal compared to their neighbors. Both contribute additional noise to extracted spectra but neither are substantially time-varying and therefore cannot be treated with time-based methods. :code:`HUSTLE-tools` offers two routines for treating hot, cold, and dead pixels: Laplacian Edge Detection and spatial smoothing.

Step 3a: Laplacian Edge Detection parameters
''''''''''''''''''''''''''''''''''''''''''''
Laplacian Edge Detection (LED) is a method of detecting defective pixels through their steep changes in flux levels relative to their neighbors. It was developed and detailed in `van Dokkum 2001 <https://iopscience.iop.org/article/10.1086/323894>`_ and we implement this method into :code:`HUSTLE-tools`. Set :code:`do_led` to True if you want to apply this routine to your G280 data. Adjust :code:`led_threshold` to set the threshold at which outliers are rejected, with lower values flagging more pixels. You can also adjust the :code:`led_factor` to change the factor by which the frames are subsampled during LED (see `van Dokkum 2001 <https://iopscience.iop.org/article/10.1086/323894>`_ for details). You can set the number of iterations LED is run for by adjusting :code:`led_n`, which may be useful for whittling down large blobs of hot, cold, and/or dead pixels since only the edge pixels will be detected on each iteration. Spectral traces are thin structures which can have sharp edges that can be mistakenly attacked by LED. To protect large fine features such as traces from excessive targeting by LED, set :code:`fine_structure` to True and adjust the :code:`contrast_factor` to tune how sensitive this model is to rejecting outliers. When :code:`fine_structure` is True, only pixels that trip both the :code:`led_threshold` and :code:`contrast_factor` thresholds will be rejected.

Step 3b: Spatial smoothing parameters
'''''''''''''''''''''''''''''''''''''
LED can be time-intensive as it requires building a noise model and optionally a fine structure model for each full frame interpolated to higher resolution. :code:`HUSTLE-tools` also includes a simple spatial median filtering method which is faster because it operates on native resolution frames and can operate on limited frame regions, but may attack fine features more aggressively. Set :code:`do_smooth` to True to apply this routine to your G280 data. select one of three :code:`smth_type` options ('1D_smooth', '2D_smooth', 'polyfit') to adjust how the median filtered model of each frame is built. The window or kernel size used for filtering is set by :code:`smth_kernel` and ideally should be similar or less than the lengthscale of your trace width. Adjust :code:`smth_threshold` to higher/lower values to flag outliers less/more aggressively. To speed reduction by focusing only on the data (particularly useful if the 'polyfit' method is used), set :code:`smth_bounds` to encompass only the traces you intend to extract from.

Step 4: Background subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The HST WFC3/UVIS G280 sky is complicated, showing variations in both space and time. The trace also occupies a substantial fraction of the frame, and the slitless nature of the G280 grism means that background objects' dispersed spectra can interfere with background estimation. To accomodate this, :code:`HUSTLE-tools` offers a wide range of background treatment options for you to explore to determine which suits your data best.

Step 4a: uniform value background subtraction
'''''''''''''''''''''''''''''''''''''''''''''
This is the simplest and fastest method of background correction, treating the background value as uniform in each frame and estimating it using the histogram of pixel flux values collected from the full frame or from a subset of the frame. However, it is also the most susceptible to errors if spatial variations in background level are present. In particular, bimodality in the background value due to contamination from the 0th order and trace wings can affect uniform background value estimation. Set :code:`do_uniform` to True to apply this method of background subtraction to your G280 data. The :code:`fit` parameter allows you to extract the mode, median, or Gaussian fit mean from the histogram of pixel flux values. The :code:`bounds` parameter can be specified to only collect pixel flux values from certain regions of the frame, reducing the impact of 0th order and trace wing flux as well as contaminating background objects. The histogram parameters, including its minimum, maximum, and number of bins spanning that range of values, can be specified by the :code:`hist_min`, :code:`hist_max`, and :code:`hist_bins` parameters.
  
Step 4b: Column-by-column background subtraction
''''''''''''''''''''''''''''''''''''''''''''''''
This method of background correction treats the value of the background as constant along columns but not along rows. While this can accomodate spatial variations more effectively than a uniform treatment, columns nearest the 0th order may struggle to acquire enough signal uncontaminated by 0th order wing flux to estimate the background accurately, meaning this method works better on redder wavelengths than on bluer wavelengths. Set :code:`do_column` to True to apply this method of background subtraction to your G280 data. You can specify which rows to draw the background estimates from using the :code:`rows` parameter, or you can set :code:`mask_trace` to True and only use pixels that are at least :code:`dist_from_trace` pixels away from the trace to estimate the background with. As column-to-column variations in background level are not expected to be very large, outlier estimates of background values can be rejected using the :code:`col_sigma` threshold.
  
Step 4c: Pagul et al. background subtraction
''''''''''''''''''''''''''''''''''''''''''''
Recently, the complex spatial structure of the G280 sky has been recognized and efforts have been taken to accurately model it. :code:`HUSTLE-tools` can fit the background in each frame by scaling the empirically-determined background sky structure image measured by `Pagul et al. 2023 <https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2023/WFC3-ISR-2023-06.pdf>`_ and hosted on `https://www.stsci.edu/hst/instrumentation/wfc3/documentation/grism-resources/uvis-grism-sky-images <https://www.stsci.edu/hst/instrumentation/wfc3/documentation/grism-resources/uvis-grism-sky-images>`_. This method of background correction is the most effective in treating spatial variations in background signal, but requires more run-time and a lot of fine-tuning to get the fit process to succeed. Set :code:`do_Pagul` to True to apply this method of background subtraction to your G280 data. Supply the absolute path of your downloaded G280 sky image to the :code:`path_to_Pagul` variable. To fit the background accurately, target and background sources in the data must be masked. Tune :code:`mask_parameter` until the diagnostic mask image output by this routine masks over sources without masking over so much of the frame that there is insufficient flux from which to estimate the background. As the median background value in each frame is not expected to vary dramatically between frames, outlier estimates of the paramter by which the empirical sky image is scaled to match the data can be rejected by setting :code:`smooth_fits` to True and adjusting :code:`smoothing_param` to higher/lower values to reject outliers less/more aggressively. Early versions of the empirical sky frame were undersampled in regions of the detector where observers frequently place their target traces. To smooth over these undersampled regions, set :code:`median_columns` to True. Later versions of the empirical sky image (e.g. v1.0 and above) generally do not need to be smoothed in this manner.

Step 5: Displacement estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Drift in the telescope pointing over the course of a visit can introduce systematic trends into the extracted 1D spectral time-series or cause offsets in the wavelength solution assigned by :code:`grismconf`. To correct for these trends, :code:`HUSTLE-tools` offers many methods of tracking telescope drift.

Step 5a: Refine target location
'''''''''''''''''''''''''''''''
This routine refines estimates of the target's location in the direct photometric image using centroiding, which can ensure a more accurate wavelength solution in Stage 2. Set :code:`do_location` to True to run this routine and update the target location for use in Stage 2.
  
Step 5b: Source center-of-mass tracking
'''''''''''''''''''''''''''''''''''''''
The saturated 0th order can be tracked via centroiding, although the bloom above and below can affect y-position determination. Set :code:`do_0thtracking` to True to run this routine. If you did not run location fitting to this point, you can manually set the :code:`location` here, which is needed to initiate the 0th centroiding routine.

Step 5c: Background star tracking
'''''''''''''''''''''''''''''''''
If there are unsaturated background stars available in your frame, they may be more useful to track than the 0th order. Set :code:`do_bkg_stars` to True to run this routine. For each background star you wish to track, supply its x and y coordinates as a list entry in the :code:`bkg_stars_loc` list.
  
Step 6: Quality quicklook
~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to save a "quicklook" gif that presents all of the data frames in succession as well as simple diagnostics of the image and trace flux, simply set :code:`do_quicklook` to True. Additionally, this will save a quicklook gif made using the data quality array, which can be checked to ensure that your reduction has not overcorrected the data.
  
Step 7: Save outputs
~~~~~~~~~~~~~~~~~~~~
If you want to save your cleaned images out to an xarray .nc file (necessary for running Stage 2), set :code:`do_save` to True.

Running Stage 1
---------------

Now that you understand what each variable does, edit your config file as you like. With the configuration file created and stored in :code:`configs/stage_1_input_config.hustle`, create a simple .py or .ipynb script with the following contents:

.. code-block:: bash

  from hustle_tools import run_pipeline
  
  config_files_dir = "configs"
  stages = (1,)
  
  run_pipeline(config_files_dir, stages)

Then execute this script to run Stage 1! The output in your cell should look similar to the output shown below, where we have used HST-GO 17183 (PI: Hannah Wakeford), visit 12, target WASP-127 as an example:

.. include:: stage_1_output.txt
   :literal:


Assessing Stage 1's success
---------------------------
Stage 1 is the most customizable stage and has a lot of diagnostics to look over. You will know if Stage 1 succeeded if:

  1. Maps of pixels flagged by temporal and spatial outlier rejection routines (e.g. CR_location.png, LED_location_of_all_corrected_pixels.png) show largely random spatial distributions with no obvious correlation to the trace. Do not worry if methods like LED targeted the 0th order bloom (large vertical spike in the middle); as long as the traces were preserved, the extraction should be reliable.
  2. The plot of estimated background values over time (bkg_values_{name of method you used}.png) shows reasonable background values consistent with the typical value of pixels away from the trace, and the plot showing the median background pixel values before and after correction (bkg_correction_{name of method you used}.png) shows that the post-corrected values are consistent with 0 e-.
  3. Measured 0th-order displacements are reasonable (should be order pixels to sub-pixels, not tens of pixels), and if available, the background star displacements are reasonably consistent with each other and the 0th-order displacements.
  4. The quicklookup.gif created by this stage shows a clear transit, a smooth background signal, no flickering or other odd visual patterns, and no obvious cosmic rays left over after cleaning.
  5. Additionally, the quicklookupDQ.gif created by this stage shows a spatially random distribution of flagged pixels - at no point should you be able to see the shape of the trace in this gif, and if you do see it, you have overcorrected the data!
