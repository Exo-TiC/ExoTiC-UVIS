Stage 0: Data Handling
======================
Prefer Jupyter notebooks? A .ipynb of this tutorial is also available on `GitHub <https://github.com/Exo-TiC/HUSTLE-tools/blob/main/tutorials/HUSTLE_tools_stage0.ipynb>`_!

The first stage of any data reduction project is to download the data. :code:`HUSTLE-tools` Stage 0 handles downloading and organizing the data for your program ID, visit number, and target of interest. The only engagement required from you is to supply the .hustle configuration file and identify the target star in the direct image. This tutorial will walk you through this process.

1. Creating the Stage 0 configuration file
---------------------------------------

The first step is to create the configuration file that will guide the execution of Stage 0. Create a folder to store the configuration file in, e.g. :code:`configs/`. Then create :code:`configs/stage_0_input_config.hustle` and populate it with the following template:

.. code-block:: bash

  # HUSTLE-tools config file for launching Stage 0: Data Handling

  # Setup for Stage 0
  toplevel_dir    './files'                                   # Directory where you want your files to be stored after Stage 0 has run. This is where /specimages, /directimages, /visitfiles, and /miscfiles will be stored.
  verbose         2                                           # Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.
  show_plots      2                                           # Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.
  save_plots      2                                           # Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.

  # Step 1: Download files from MAST
  do_download     True                                        # Bool. Whether to perform this step.
  programID       '12345'                                     # ID of the observing program you want to query data from. On MAST, referred to as "proposal_ID".
  target_name     'PLANET-B'                                  # Name of the target object you want to query data from. On MAST, referred to as "target_name".
  token           None                                        # str or None. If you are downloading proprietary data, please visit https://auth.mast.stsci.edu/token?suggested_name=Astroquery&suggested_scope=mast:exclusive_access to obtain an authentication token and enter it as a '' string here.
  extensions      ['_flt.fits', '_jit.fits']                  # lst of str or None. File extensions you want to download. If None, take all file extensions. Otherwise, take only the files specified. _flt.fits are required. _jit.fits are recommended if you want to use jitter decorrelation to detrend systematics.

  # Step 2: Organizing files
  do_organize     True                                        # Bool. Whether to perform this step.
  visit_number    '00'                                        # The visit number you want to operate on.
  filesfrom_dir   None                                        # None or str. If you downloaded data in Step 1, leave this as None. If you have pre-downloaded data, please place all of it in filesfrom_dir. Don't sort it into sub-folders; HUSTLE-tools won't be able to find them if they are inside sub-folders!

  # Step 3: Locating the target star
  do_locate       True                                        # Bool. Whether to perform this step.
  location        None                                        # None or tuple of float. Prior to running Stage 0, this will be None. After running Stage 0, a copy of this .hustle file will be made with this information included.

  # Step 4: Quality quicklook
  do_quicklook    True                                        # Bool. Whether to perform this step.
  traces_included ('+1',)                                     # List of str. Which traces are included in the white light curve plot included in the quicklook.

  # ENDPARSE

Let's break down each of these steps to make sure we understand what we can tune in this stage.

Setup for Stage 0
~~~~~~~~~~~~~~~~~
You can customize the :code:`toplevel_dir` folder to be any name you like as long as you use the same folder name for all stages. The :code:`verbose`, :code:`show_plots`, and :code:`save_plots` keys respectively control the level of detail in the statements printed by the pipeline while it executes, the number of plots opened in the terminal or .ipynb notebook cell you are running the script in, and the number of plots saved out to .png or .gif files to be reviewed at any time after execution.

Step 1: Download files from MAST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have not pre-downloaded the data for your program, then set :code:`do_download` to True. If you are unsure of your obsevation's :code:`programID` and/or :code:`target_name`, you can head to `https://www.stsci.edu/hst/observing/program-information <https://www.stsci.edu/hst/observing/program-information>`_ and search for your program by proposal ID and/or PI/Co-I name. If your data is protected by a proprietary period, you can visit `https://auth.mast.stsci.edu/token?suggested_name=Astroquery&suggested_scope=mast:exclusive_access <https://auth.mast.stsci.edu/token?suggested_name=Astroquery&suggested_scope=mast:exclusive_access>`_ to acquire an authentication token, which you supply as a string encased in '' marks to the :code:`token` variable. The :code:`extensions` variable controls which file extensions you want to acquire. The minimum recommended case includes all images, while additional extensions may be useful to acquire e.g. telescope telemetry data.

Step 2: Organizing files
~~~~~~~~~~~~~~~~~~~~~~~~
If you have not pre-downloaded the data, or if you have pre-downloaded the data but they are not yet organized, then set :code:`do_organize` to True. This step will sort your files by orbit number, visit number, and file contents. Specify the :code:`visit_number` variable as a two-character string to select which visit in the observation corresponds to your target. As before, you can find the visit number by heading to `https://www.stsci.edu/hst/observing/program-information <https://www.stsci.edu/hst/observing/program-information>`_ and searching for your program by proposal ID and/or PI/Co-I name. If you have pre-downloaded your data, store all of it in the folder specified in the :code:`filesfrom_dir` variable.

Step 3: Locating the target star
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set :code:`do_locate` to True to prompt the pipeline to present the direct image so that you can locate the target star within this frame. This is the only step of the pipeline which requires user input, but it is necessary to ensure successful wavelength calibration in Stage 2. If you have already located the coordinates of the target star, you can set :code:`do_locate` to False and simply supply the coordinates as a tuple of floats to the :code:`location` variable. Otherwise, a copy of this configuration file will be output after execution which will update the :code:`location` variable to the coordinates selected during the :code:`do_locate` step.

Step 4: Quality quicklook
~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to save a "quicklook" gif that presents all of the data frames in succession as well as simple diagnostics of the image and trace flux, simply set :code:`do_quicklook` to True. You can use the :code:`traces_included` variable to include flux from the +1 trace, the -1 trace, or both.

Running Stage 0
---------------

Now that you understand what each variable does, edit your config file as you like. With the configuration file created and stored in :code:`configs/stage_0_input_config.hustle`, create a simple .py or .ipynb script with the following contents:

.. code-block:: bash

  from hustle_tools import run_pipeline
  
  config_files_dir = "configs"
  stages = (0,)
  
  run_pipeline(config_files_dir, stages)

Then execute this script to run Stage 0! The output in your cell should look similar to the output shown below, where we have used HST-GO 17183 (PI: Hannah Wakeford), visit 12, target WASP-127 as an example:


.. include:: stage_0_output.txt
   :literal:


Assessing Stage 0's success
---------------------------
Stage 0 is the simplest stage that has very few diagnostics to look over. You will know if Stage 0 succeeded if:

  1. The :code:`toplevel_dir` folder has been created and populated with the :code:`specimages`, :code:`directimages`, :code:`jitterfiles`, :code:`visitfiles`, :code:`miscfiles`, and :code:`outputs` subfolders.
  2. The :code:`toplevel_dir/outputs/stage_0` folder contains an updated copy of the .hustle configuration folder with the :code:`location` variable changed from None to a tuple of floats.
  3. The quicklookup.gif created by this stage, or the .fits files downloaded to the :code:`toplevel_dir/specimages` directory, clearly show your target star and contain all of the orbits and total number of frames you expected.
