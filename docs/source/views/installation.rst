Installation
============


Step 1: Install HUSTLE-tools from GitHub
________________________________________

Currently :code:`HUSTLE-tools` is only available to download or clone from GitHub. You can download :code:`HUSTLE-tools` from `GitHub <https://github.com/Exo-TiC/HUSTLE-tools>`_ or you can clone the repository:

.. code-block:: bash

  git clone https://github.com/Exo-TiC/HUSTLE-tools

Then navigate into the top level of the newly-cloned repository and install:

.. code-block:: bash

  cd HUSTLE-tools
  pip install .

Step 2: Download supporting files for :code:`grismconf`
_______________________________________________________

:code:`HUSTLE-tools` uses :code:`grismconf` (hosted on GitHub at `this link <https://github.com/npirzkal/GRISMCONF>`_ and developed by `Pirzkal & Ryan 2017 <https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2017/WFC3-2017-01.pdf>`_) to assign wavelength solutions to G280 exposures. To use :code:`grismconf` you must download the associated `WFC3/UVIS configuration files <https://github.com/npirzkal/GRISM_WFC3>`_ and supply the absolute path to these files on your computer in the Stage 2 .hustle files. Make sure to download these if you intend to run Stage 2!

Step 3 (optional): Download WFC3/UVIS sky background files
__________________________________________________________

The WFC3/UVIS G280 has a unique background pattern owing in part to the amplifier split midway across the detector (see `Pagul et al. 2023 <https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2023/WFC3-ISR-2023-06.pdf>`_ for details). :code:`HUSTLE-tools` uses the empirically-determined G280 sky image in a custom background correction routine contained in Step 5c of Stage 1. If you want to make use of this routine, simply head over to `https://www.stsci.edu/hst/instrumentation/wfc3/documentation/grism-resources/uvis-grism-sky-images <https://www.stsci.edu/hst/instrumentation/wfc3/documentation/grism-resources/uvis-grism-sky-images>`_ and download the latest type FLT sky image appropriate to the chip used in your observations (typically chip 2). Then supply the absolute path to this file on your computer in the Stage 1 .hustle file.
