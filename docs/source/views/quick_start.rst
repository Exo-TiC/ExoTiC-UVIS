Quick start
===========

Ready to get going with :code:`HUSTLE-tools`? This quickstart guide will help you understand the basics of running :code:`HUSTLE-tools` on your G280 data. More detailed instructions for each Stage can be found in the :ref:`Tutorials` page.

1. Install :code:`HUSTLE-tools`
-------------------------------

The first step to running :code:`HUSTLE-tools` is to make sure you have it and its dependencies installed. Follow the instructions on the :ref:`Installation` page to get your :code:`HUSTLE-tools` conda environment set up and ready to go.

2. Set up a run directory
-------------------------

We recommend keeping your G280 data reduction projects separate for ease of navigation. For this quickstart, let's create a directory for the observations of HAT-P-41B from visit 01 of HST-GO 15288 (PI: David Sing).:

.. code-block:: bash

  mkdir /User/hustle-tools_demo/
  cd /User/hustle-tools_demo/

:code:`HUSTLE-tools` operates by reading in .hustle configuration files to a high-level wrapper function, :code:`hustle_tools.run_pipeline()`. To run :code:`HUSTLE-tools` in our new run directory, we'll need to create two items:
  1. A folder to store our .hustle configuration files in.
  2. A simple .py script to run the pipeline with.

.. code-block:: bash

  mkdir configs
  vim run_pipeline.py

We'll need to populate each of these items as follows.

2.1. Create the pipeline wrapper script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pipeline wrapper script is very simple; it just needs to know what stages you want to run and where the configuration files are being stored. Using your favorite text editor, write the following script into :code:`run_pipeline.py`:

.. code-block:: bash

  from hustle_tools import run_pipeline
  
  config_files_dir = "configs"
  stages = (0,1,2)
  
  run_pipeline(config_files_dir, stages)

That's it!

2.2. Supply the configuration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The .hustle configuration files are at the core of operating :code:`HUSTLE-tools` and they take some time to get to know. For this quickstart, we've written the .hustle files for you, but if you want to learn more about how to tune these files for your needs, check out the :ref:`Tutorials` tab! For now, just :code:`cd` into :code:`configs` and follow the instructions below to create the .hustle configuration files for this run.

First, use your favorite text editor to create :code:`configs/stage_0_input_config.hustle` and populate it with the following script:

.. code-block:: bash

  """i will come back to this later i promise :3"""

Next, create :code:`configs/stage_1_input_config.hustle` and populate it with the following script:

.. code-block:: bash

  """this too :3"""

Lastly, create :code:`configs/stage_2_input_config.hustle` and populate it with the following script:

.. code-block:: bash

  """and this :3"""

3. Run :code:`HUSTLE-tools`
--------------------------

You are now ready to run :code:`HUSTLE-tools`!

.. code-block::

  cd ..
  python run_pipeline.py

Most of the pipeline will run hands-free. However, in Stage 0 you will be presented with the direct photometric image taken as part of these observations and asked to locate the target star in the image, which is essential to getting the wavelength solution right. In these observations, you will find the target star at :code:`x=???` and :code:`y=???`. After this step, the pipeline will operate on its own. On an average laptop, it shouldn't take more than five minutes

4. Examine the outputs
----------------------

If you reached the end with no errors, congratulations! You have successfully run :code:`HUSTLE-tools`.
