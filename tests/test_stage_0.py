import os
import shutil
import unittest
import numpy as np
from astroquery.mast import Observations

from exotic_uvis import stage_0


class TestStage0(unittest.TestCase):
    """ Test exotic_uvis stage 0. """

    @classmethod
    def setUpClass(cls):
        # Define local test path, and clear cached data.
        cls.local_data_path = 'test_exotic_uvis_data'
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)

        # Download lightweight test data.
        # First two exposures of first two orbits of W31, proposal_id=17183.
        cls.mast_data_files = [
            "iexr16ljq_flt.fits",  # Direct images.
            "iexr16ljq_spt.fits",
            "iexr16lkq_flt.fits",  # Spec images.
            "iexr16lkq_spt.fits",
            "iexr16llq_flt.fits",
            "iexr16llq_spt.fits",
            "iexr16luq_flt.fits",
            "iexr16luq_spt.fits",
            "iexr16lvq_flt.fits",
            "iexr16lvq_spt.fits"]
        os.mkdir(cls.local_data_path)
        for mdf in cls.mast_data_files:
            Observations.download_file(
                "mast:HST/product/{}".format(mdf),
                local_path=os.path.join(cls.local_data_path, mdf))

    @classmethod
    def tearDownClass(cls):
        # Tidy up.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)

    def test_download_files_from_mast(self):
        """ Test download files from MAST. """
        for mdf in self.mast_data_files:
            self.assertTrue(os.path.exists(os.path.join(
                self.local_data_path, mdf)))


if __name__ == '__main__':
    unittest.main()
