import os
import shutil
from urllib import request
import unittest
from astroquery.mast import Observations
from astropy.io import fits
import numpy as np
from scipy.signal import medfilt2d
import xarray as xr

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

        # Download the UVIS GRISMCONF config.
        config_path = os.path.join(cls.local_data_path, "UVIS_conf")
        os.mkdir(config_path)
        request.urlretrieve("https://raw.githubusercontent.com/npirzkal/GRISM_WFC3/refs/heads/master/UVIS/UVIS_G280_CCD2_V2.conf",
                            os.path.join(config_path,"UVIS_G280_CCD2_V2.conf"))
        cls.path_to_cal = os.path.join(config_path,"UVIS_G280_CCD2_V2.conf")
        for i in range(1,5):
            request.urlretrieve("https://github.com/npirzkal/GRISM_WFC3/raw/refs/heads/master/UVIS/WFC3.UVIS.G280.CCD2.p{}.sens.2021.fits".format(i),
                                os.path.join(config_path,"WFC3.UVIS.G280.CCD2.p{}.sens.2021.fits".format(i)))
        for i in range(1,5):
            request.urlretrieve("https://github.com/npirzkal/GRISM_WFC3/raw/refs/heads/master/UVIS/WFC3.UVIS.G280.CCD2.m{}.sens.2021.fits".format(i),
                                os.path.join(config_path,"WFC3.UVIS.G280.CCD2.m{}.sens.2021.fits".format(i)))
        request.urlretrieve("https://github.com/npirzkal/GRISM_WFC3/raw/refs/heads/master/UVIS/WFC3.UVIS.G280.CCD2.0.sens.2021.fits",
                            os.path.join(config_path,"WFC3.UVIS.G280.CCD2.0.sens.2021.fits"))

        # Invent a clean_obs.nc to read.
        # First, use a single spec image from the test data as a template.
        template = cls.xarray_data.images.data[0,:,:]

        # GREATLY simplify the data.
        print("Simplifying template data...")
        for i in range(15):
            template = medfilt2d(template,kernel_size=5) # aggressively smooth it
        template -= np.min(template) # normalize
        template /= np.max(template[template<35000]) # need to ignore the highly saturated 0th

        # Copy it into a 20-frame data set.
        print("Extending in time and adding a 1% transit...")
        x,y = template.shape
        images = np.empty((20,x,y))
        for k in range(20):
            images[k,:,:] = template
            if (k > 5 and k < 15):
                images[k,:,:] *= 0.99 # a 1% transit occurs!

        # Add a tiny smattering of noise.
        print("Adding a very small noise component...")
        errors = np.empty_like(images)
        np.random.seed(1337) # to keep noise draws consistent, apply a seed.
        for k in range(20):
            error = np.abs(np.random.normal(0.0,1e-3,(x,y)))
            errors[k,:,:] = error
            images[k,:,:] += error # noise is of order 1000 ppm, transit depth is 10,000 ppm.

        # Create x-array
        obs = xr.Dataset(
            data_vars=dict(
                images=(["exp_time", "x", "y"], images),
                errors=(["exp_time", "x", "y"], errors),
                subarr_coords=(["index"],cls.xarray_data.subarr_coords.values),
            ),
            coords=dict(
                exp_time=np.array([i for i in range(20)]),
                index=["left edge", "right edge", "bottom edge", "top edge"],
            ),
            attrs = dict(
                target_posx = 968, # known from looking at file
                target_posy = 170,
            )
        )
        # Replace class xarray_data with our new model.
        cls.xarray_data = obs

        # Save it out.
        stage_1.save_data_S1(obs,cls.local_data_path)
        print("Test data is ready to go!")

    @classmethod
    def tearDownClass(cls):
        # Tidy up.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)
    
    def test_a_load_data(self):
        """ Load .fits files as xarray."""
        obs = stage_2.load_data_S2(self.local_data_path)
        self.assertEqual(obs.images.shape[0], 20)

    def test_b_get_trace_solution(self):
        """ Get the GRISMCONF solution to the trace position. """
        # We only downloaded the config for the +1 order, so we fit for that.
        pos = [self.xarray_data.attrs['target_posx'],
               self.xarray_data.attrs['target_posy']]
        
        # We'll need this info for the next test, so globalize it.
        global trace_x, trace_y, wavs
        trace_x, trace_y, wavs, widths, sens = stage_2.get_trace_solution(self.xarray_data,
                                                                          order="+1",
                                                                          source_pos=pos,
                                                                          refine_calibration=False,
                                                                          path_to_cal=self.path_to_cal,
                                                                          verbose=2)
        # The +1 order should span 2000 to 8000 AA
        # and the shapes of trace_x and y should match,
        # except trace_y has a time component.
        self.assertTrue(np.min(wavs)>=2000)
        self.assertTrue(np.max(wavs)<=8000)
        self.assertEqual(trace_x.shape[0],trace_y.shape[1])
        self.assertEqual(trace_y.shape[0],20)

    def test_c_determine_ideal_halfwidth(self):
        """ Use out-of-transit residuals to optimize the halfwidth. """
        # Same as before, going to need this later.
        global halfwidth
        halfwidth = stage_2.determine_ideal_halfwidth(self.xarray_data,
                                                      order="+1",
                                                      trace_x=trace_x,
                                                      trace_y=trace_y,
                                                      wavs=wavs,
                                                      indices=[[0,5],[15,19]],
                                                      verbose=2,
                                                      save_plots=2,
                                                      output_dir='./')
        # With the fixed noise draw we had, it usually found hw 18.
        # I'll accept anywhere in the range 15 to 21 honestly.
        self.assertAlmostEqual(halfwidth,18,delta=3)

    def test_d_extraction(self):
        """ Standard and Horne 1986 optimal 1D spectral extraction methods. """
        # First, a standard extraction with no weighting.
        global spec, spec_err
        spec, spec_err = stage_2.standard_extraction(self.xarray_data,
                                                     halfwidth=halfwidth,
                                                     trace_x=trace_x,
                                                     trace_y=trace_y,
                                                     order="+1",
                                                     masks=[[100,100,20]],
                                                     verbose=2)
        
        # The spec and spec_err should come out to the same shape.
        self.assertEqual(spec.shape,spec_err.shape)

        # Because of how we defined the errors array,
        # spec should be bigger than spec_err.
        self.assertTrue((spec_err<spec).all())

    def test_e_align(self):
        """ Cross correlation to correct for position shifts. """
        # We did not add any shifts to our fake data, so all shifts should be ~0.
        aligned_spec, aligned_spec_err, shifts = stage_2.align_spectra(self.xarray_data,
                                                                       np.array(spec),
                                                                       np.array(spec_err),
                                                                       order="+1",
                                                                       trace_x=np.array(wavs),
                                                                       align=True,ind1=10,ind2=-10,
                                                                       verbose=2)
        # I will tolerate a 1/10 pixel shift at most, given how small the
        # added noise component was.
        self.assertEqual(len(shifts),20)
        for shift in shifts:
            self.assertTrue(abs(shift)<0.10)

    def test_f_clean(self):
        """ Removal of temporal outliers from spectra. """
        # Let's add an outlier to clean in the mid-transit.
        spec[10,200] = 1e12
        original_spec = np.copy(spec)
        specs = stage_2.clean_spectra([spec,],
                                      sigma=4.0)
        
        # Check how many outliers were found.
        S = np.count_nonzero(np.where(specs[0]!=original_spec, 1, 0))
        self.assertEqual(S,1,
                         msg='Cleaning failed!\nNumber of outliers found: {}\n Number expected: 1'.format(S))

if __name__ == '__main__':
    unittest.main()