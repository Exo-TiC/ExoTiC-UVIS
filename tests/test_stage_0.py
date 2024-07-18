import os
import shutil
import unittest
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

    def test_organise_downloaded_files(self):
        """ Collect, move, and label downloaded files. """
        spec_flt, spec_spt, direct_flt, direct_spt, misc_files = \
            stage_0.collect_files(self.local_data_path)
        self.assertEqual(len(spec_flt), 4)
        self.assertEqual(len(spec_spt), 4)
        self.assertEqual(len(direct_flt), 1)
        self.assertEqual(len(direct_spt), 1)
        self.assertEqual(len(misc_files), 0)

        stage_0.identify_orbits(spec_flt, spec_spt, direct_flt, direct_spt, misc_files)
        stage_0.move_files(self.local_data_path, self.local_data_path)
        self.assertTrue(os.path.exists(os.path.join(
            self.local_data_path, "directimages", "or01dr001_flt.fits")))
        for orbit in ["01", "02"]:
            for exposure in ["001", "002"]:
                for f_type in ["flt", "spt"]:
                    self.assertTrue(os.path.exists(os.path.join(
                        self.local_data_path, "specimages", "or{}fm{}_{}.fits"
                            .format(orbit, exposure, f_type))))


if __name__ == '__main__':
    unittest.main()
