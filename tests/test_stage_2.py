import os
import shutil
from urllib import request
import unittest
from astroquery.mast import Observations
from astropy.io import fits
import numpy as np

from exotic_uvis import stage_0, stage_1, stage_2

class TestStage2(unittest.TestCase):
    """ Test exotic_uvis stage 2. """

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
        stage_0.collect_and_move_files(
            cls.visit_number, cls.local_data_path, cls.local_data_path)
        cls.xarray_data = stage_1.load_data_S1(cls.local_data_path)

        config_path = os.path.join(cls.local_data_path, "G280_config")
        os.mkdir(config_path)
        request.urlretrieve("https://www.stsci.edu/~WFC3/grism-resources/uvis-grism-sky-images/sky_G280_chip2_flt_hdr.fits",
                            os.path.join(config_path,"Pagul_sky.fits"))
        cls.Pagul_path = os.path.join(config_path,"Pagul_sky.fits")

    @classmethod
    def tearDownClass(cls):
        # Tidy up.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)
    
    def test_load_data(self):
        """ Load .fits files as xarray."""
        obs = stage_1.load_data_S1(self.local_data_path)
        self.assertEqual(obs.images.shape[0], 4)


    def test_temporal_outlier_rejection(self):
        """ Replace cosmic rays with various methods. """
        stage_1.fixed_iteration_rejection(self.xarray_data, sigmas=[5,5,5], replacement=None)

        stage_1.free_iteration_rejection(self.xarray_data, threshold=3.5, verbose_plots=0,
                                               check_all=False, output_dir=None)
        
        # obs = stage_1.sigma_clip_rejection(self.xarray_data, sigma=5)

    def test_spatial_outlier_rejection(self):
        """ Clean hot pixels with various methods. """
        stage_1.laplacian_edge_detection(self.xarray_data, sigma=10, factor=2,
                                               n = 2, build_fine_structure=True, contrast_factor=5)
        
        stage_1.spatial_smoothing(self.xarray_data, sigma=10)

    def test_bckg_subtract(self):
        """ Subtract background signal with various methods. """
        obs, A, modes = stage_1.Pagul_bckg_subtraction(self.xarray_data,
                                                       Pagul_path = self.Pagul_path,
                                                       masking_parameter=1,
                                                       median_on_columns=False)
        self.assertEqual(obs.images.shape[0],len(A))

        obs, bckgs = stage_1.full_frame_bckg_subtraction(self.xarray_data, bin_number=1)
        self.assertEqual(obs.images.shape[0], len(bckgs))

if __name__ == '__main__':
    unittest.main()