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

        # Download lightweight test data. First two exposures of first two
        # orbits of W31, proposal_id=17183, visit=16.
        cls.visit_number = "16"
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
    
    def test_a_download_files_from_mast(self):
        """ Test download files from MAST. """
        for mdf in self.mast_data_files:
            self.assertTrue(os.path.exists(os.path.join(
                self.local_data_path, mdf)))
    
    def test_b_organise_downloaded_files(self):
        """ Collect, move, and label downloaded files. """
        stage_0.collect_and_move_files(
            self.visit_number, self.local_data_path, self.local_data_path)

        self.assertTrue(os.path.exists(os.path.join(
            self.local_data_path, "directimages", "or01dr001_flt.fits")))
        for orbit in ["01", "02"]:
            for exposure in ["001", "002"]:
                for f_type in ["flt", "spt"]:
                    self.assertTrue(os.path.exists(os.path.join(
                        self.local_data_path, "specimages", "or{}fm{}_{}.fits"
                            .format(orbit, exposure, f_type))))
    
    def test_c_locate_target(self):
        """ Locate source in direct image. """
        source_x, source_y = stage_0.locate_target('none',
            test=True)
        self.assertEqual(source_x,50)
        self.assertEqual(source_y,50)


if __name__ == '__main__':
    unittest.main()
