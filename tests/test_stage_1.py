import os
import shutil
from urllib import request
import unittest
from astroquery.mast import Observations
from astropy.io import fits
import numpy as np
import xarray as xr

from exotic_uvis import stage_0, stage_1

class TestStage1(unittest.TestCase):
    """ Test exotic_uvis stage 1. """

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

        # Modify loaded data to be easy to test on.
        # Create a template image which is the Pagul image with a star in the middle.
        print("Replacing loaded data with the Pagul+ 2023 sky image...")
        x1,x2,y1,y2 = [int(x) for x in cls.xarray_data.subarr_coords.values]
        with fits.open(cls.Pagul_path) as fits_file:
            pagul_star = fits_file[0].data[y1:y2+1,x1:x2+1]
            x0, y0 = pagul_star.shape
            x0 /= 2
            y0 /= 2
            for i in range(int(x0)-50,int(x0)+50):
                for j in range(int(y0)-50,int(y0)+50):
                    v = 100*np.exp(-((i-x0)**2 + (j-y0)**2)/50)
                    pagul_star[i,j] += v
        cls.star_truth = (y0,x0)

        # Replace the loaded data with our fake data.
        # It's embedded to replicate real data conditions.
        cls.xarray_data.images.data[:,:,:] = np.zeros_like(cls.xarray_data.images.data[:,:,:])
        for k in range(4):
            cls.xarray_data.images.data[k,:,:] = pagul_star
        # Replace the direct image too.
        cls.xarray_data.direct_image.data = pagul_star
        # Zero out the data quality array.
        cls.xarray_data.data_quality.data[:,:,:] = np.zeros_like(cls.xarray_data.data_quality.data[:,:,:])

        # We need to have a background star as well, for tracking.
        # It wiggles slightly so we have motion to track.
        print("Adding a background star to the images...")
        x0 = 1625
        y0 = 500
        cls.bkg_truth = (y0,x0)
        off1s = [0 if i%2==0 else -1 for i in range(4)]
        off2s = [0 if i%2==0 else -1 for i in range(4)]
        for k, (off1,off2) in enumerate(zip(off1s,off2s)):
            for j in range(int(x0+off1)-50,int(x0+off1)+50):
                for i in range(int(y0+off2)-50,int(y0+off2)+50):
                    v = 50*np.exp(-((j-(x0+off1))**2 + (i-(y0+off2))**2)/20)
                    cls.xarray_data.images.data[k,i,j] += v
        
        # Place a series of 9 hot pixels in every image for a total of 36 hot pixels.
        print("Adding 9 hot pixels to each image...")
        for j in (250,1250,1750):
            for i in (150,250,350):
                cls.xarray_data.images.data[:,i,j] = 1e12

        # Sprinkle a tiny amount of noise in.
        print("Adding a very small noise component...")
        np.random.seed(1337) # to keep noise draws consistent, apply a seed.
        for k in range(4):
            cls.xarray_data.images.data[k,:,:] += np.random.normal(0.0,0.000001,(600,2100))
        
        # The time data needs to be different. You can't do time iteration in four frames.
        print("Creating time series to test cleaning...")
        exp_time = np.array([k for k in range(1000)]) # 1000 time stamps
        images = np.zeros((1000,10,10))
        for k in range(1000):
            images[k,:,:] += np.random.normal(1.0,0.1,(10,10)) # the data is about 1.0
        print("Adding 9 cosmic rays to the time series...")
        # Place a series of 9 cosmic rays in the 500th image.
        for j in (1,3,7):
            for i in (2,4,5):
                images[499,i,j] = 1e12 # gigantic cosmic ray can't be missed
        data_quality = np.zeros_like(images) # and a fake DQ array that is empty
        cls.xarray_tseries = xr.Dataset(data_vars=dict(
                            images=(["exp_time", "x", "y"], images),
                            data_quality = (["exp_time", "x", "y"], data_quality)),
                            coords=dict(exp_time=exp_time,),
        )

        print("Test data and time series with 45 bad pixels is now ready to go!")

    @classmethod
    def tearDownClass(cls):
        # Tidy up.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)
    
    def test_a_load_data(self):
        """ Load .fits files as xarray."""
        obs = stage_1.load_data_S1(self.local_data_path)
        # Confirm it loaded correctly by checking the shape of the images.
        self.assertEqual(obs.images.shape[0], 4)

    def test_b_temporal_outlier_rejection(self):
        """ Replace cosmic rays with various methods. """
        # For time cleaning, we supply the iterators with a tiny dataset that
        # has 9 very obvious cosmic rays in the 500th frame.
        stage_1.fixed_iteration_rejection(self.xarray_tseries, sigmas=[10,10],
                                          replacement=None, verbose=2)

        stage_1.free_iteration_rejection(self.xarray_tseries,
                                         threshold=10, verbose=2)
        
        # obs = stage_1.sigma_clip_rejection(self.xarray_data, sigma=5)

    def test_c_spatial_outlier_rejection(self):
        """ Clean hot pixels with various methods. """
        # LED should catch all the hot pixels since they are so very hot.
        stage_1.laplacian_edge_detection(self.xarray_data, sigma=10, factor=2,
                                         n=1, build_fine_structure=False,
                                         contrast_factor=5, verbose=2)
        
        # WIP!
        #stage_1.spatial_smoothing(self.xarray_data, sigma=10)
        #self.assertEqual(np.count_nonzero(self.xarray_data.data_quality.data),18)
    
    def test_d_confirm_cleaned_correctly(self):
        # After all this cleaning, 9 CRs + 36 HPs = 45 bad pixels should have been found.

        # For the test data, we check that the 36 hot pixels were flagged in the DQ array.
        space_cleaned = np.count_nonzero(self.xarray_data.data_quality.data)
        self.assertEqual(space_cleaned,36,
                         msg='Spatial cleaning failed!\nNumber of cleaned pixels: {}\n Number expected: 36'.format(space_cleaned))
        
        # For the test timeseries, we check that the 9 CRs valued at 1e12 are gone.
        CRs_left = (np.abs(self.xarray_tseries.images.data)>1e3).sum()
        self.assertEqual(CRs_left,0,
                         msg='Time cleaning failed!\nNumber of cosmic rays left: {}\n Number expected: 0'.format(CRs_left))

    def test_e_bckg_subtract(self):
        """ Subtract background signal with various methods. """
        # First we are using the Pagul+ method, since we based our 'data' on it.
        stage_1.Pagul_bckg_subtraction(self.xarray_data,
                                       Pagul_path = self.Pagul_path,
                                       masking_parameter=0.0001,
                                       median_on_columns=False)
        
        # Most of the background should be close to zero now. Check that >99% of the data is small!
        check = (np.abs(self.xarray_data.images.data)<1e-5).sum()
        self.assertTrue(check > 0.99*4*600*2100,
                        msg='Background subtraction failed!\nZeroed pixels: {:.2E} out of {:.2E} threshold for success'.format(check,0.99*4*600*2100))

        # Then we're trying uniform without bounds.
        # We know a priori our bckg is generally between -1 and 1.
        stage_1.uniform_value_bkg_subtraction(self.xarray_data, fit=None,
                                              hist_min=-1,hist_max=1,
                                              bounds=None, hist_bins=10000,
                                              verbose=2)
        
        # Since we already subtracted and zeroed out most of the data,
        # the bkg vals should be very small.
        for bkg_val in self.xarray_data['bkg_vals'].data:
            self.assertTrue(abs(bkg_val) < 1e-5)

        # Now we use the first and last four rows to get column bckg.
        stage_1.column_by_column_subtraction(self.xarray_data,
                                             rows=[0,1,2,3,-4,-3,-2,-1,],
                                             sigma=3,mask_trace=False,
                                             verbose=2)
        
        # Again, should be very small. Now the bckgs are 2D though!

        # First, make sure it's the right shape.
        self.assertTrue(self.xarray_data['bkg_vals'].data.shape == (4,2100))

        # Then check each value for smallness.
        for i in range(4):
            for j in range(2100):
                self.assertTrue(abs(self.xarray_data['bkg_vals'].data[i,j])<1e-5)
    
    def test_f_tracking(self):
        """ Tracking of source and background stars. """
        # First, we refine the direct image location.
        stage_1.refine_location(self.xarray_data,location=self.star_truth)
        for truth, retrieved in zip(self.star_truth,
                                    (self.xarray_data.attrs['target_posx'],
                                     self.xarray_data.attrs['target_posy'])):
            self.assertAlmostEqual(truth,retrieved,delta=0.10)

        # Then we track the 0th order in each frame.
        # We did not add any 0th order motion so there should be little variation.
        xs, ys = stage_1.track_0thOrder(self.xarray_data,guess=[i for i in self.star_truth])
        for x,y in zip(xs,ys):
            self.assertAlmostEqual(self.star_truth[0],x,delta=0.01)
            self.assertAlmostEqual(self.star_truth[1],y,delta=0.01)

        # Then we track our wiggling bkg star relative to its initial position.
        stars_pos, mean_pos = stage_1.track_bkgstars(self.xarray_data,
                                                     bkg_stars=([i for i in list(reversed(self.bkg_truth))],))
        
        # If the tracking worked, stars_pos[0] should alternate
        # between [0,0] and [-1,-1].
        x_retrieved = [x[0] for x in stars_pos[0]]
        x_truth = [0,-1,0,-1]
        y_retrieved = [x[1] for x in stars_pos[0]]
        y_truth = [0,-1,0,-1]
        for xi, xt, yi, yt in zip(x_retrieved,x_truth,
                                  y_retrieved,y_truth):
            self.assertAlmostEqual(xi,xt,delta=0.01)
            self.assertAlmostEqual(yi,yt,delta=0.01)

if __name__ == '__main__':
    unittest.main()