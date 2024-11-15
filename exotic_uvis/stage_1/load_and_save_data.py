import os
from tqdm import tqdm

import numpy as np
from astropy.io import fits
import xarray as xr

from wfc3tools import sub2full


def load_data_S1(data_dir, skip_first_fm = False, skip_first_or = False, verbose = 2):
    """Function to load the data into an xarray.

    Args:
        data_dir (str): folder where the spec and direct image subfolders are.
        skip_first_fm (bool, optional): whether to remove all first frames from
        each orbit. Defaults to False.
        skip_first_or (bool, optional): whether to remove the first orbit from
        the dataset. Defaults to False.
        verbose (int, optional): How detailed the print statements should be
        on a scale of 0-2. Defaults to 2.

    Returns:
        xarray: images and all associated data needed for reduction.
    """

    # initialize data structures
    images, errors, data_quality, subarr_coords = [], [], [], []
    exp_time, exp_time_UT, exp_duration, read_noise = [], [], [], []
    
    # iterate over all files in specs directory
    specs_dir = os.path.join(data_dir, 'specimages/')
    files = np.sort(os.listdir(specs_dir))

    
    for filename in tqdm(files, 'Loading data... Progress:', disable = verbose == 0):
   
        # only open flt files 
        if filename[-9:] == '_flt.fits':
            if (skip_first_fm and 'fm001' in filename):
                # skip loading first frame
                continue

            if (skip_first_or and 'or01' in filename):
                # skip loading first orbit
                continue

            # open file and save image and error
            with fits.open(os.path.join(specs_dir, filename)) as hdul:
                image = np.array(hdul[1].data)
                error = np.array(hdul[2].data)

                #print(repr(hdul[0].header))
                exp_time.append((hdul[0].header['EXPSTART'] + hdul[0].header['EXPEND'])/2)
                exp_time_UT.append((hdul[0].header['TIME-OBS']))
                data_quality.append(hdul[3].data)
                exp_duration.append(hdul[0].header["EXPTIME"])

                #run file through sub2full
                y1,y2,x1,x2 = sub2full(os.path.join(specs_dir, filename), fullExtent=True)[0]
                
                # append data
                images.append(image) 
                errors.append(error) 
                read_noise.append(np.median(np.sqrt(error**2 - image))) 
                subarr_coords.append(np.array([y1,y2,x1,x2]))

    # collapse subarr_coords
    subarr_coords = np.mean(np.array(subarr_coords),axis=0)

    # iterate over all files in direct images directory
    directimages_dir = os.path.join(data_dir, 'directimages/')

    for filename in np.sort(os.listdir(directimages_dir)):

        if filename[-9:] == '_flt.fits':

            with fits.open(os.path.join(directimages_dir, filename)) as hdul:
                image = np.array(hdul[1].data)
                
                direct_image = image
                target_posx = (direct_image.shape[1])/2 - hdul[0].header['POSTARG1']
                target_posy = (direct_image.shape[0])/2 - hdul[0].header['POSTARG2']


    # create x-array
    obs = xr.Dataset(
        data_vars=dict(
            images=(["exp_time", "x", "y"], images),
            errors=(["exp_time", "x", "y"], errors),
            subarr_coords=(["index"],subarr_coords),
            direct_image = (["x", "y"], direct_image),
            badpix_mask = (["exp_time", "x", "y"], np.ones_like(images, dtype = 'bool')),
            data_quality = (["exp_time", "x", "y"], data_quality),
            read_noise = (['exp_time'], read_noise)
        ),
        coords=dict(
            exp_time=exp_time,
            exp_time_UT = (["exp_time"], exp_time_UT),
            index=["left edge", "right edge", "bottom edge", "top edge"],
        ),
        attrs = dict(
            target_posx = target_posx,
            target_posy = target_posy,
        )
    )

    return obs


def save_data_S1(obs, output_dir, filename = 'clean_obs'):
    """Function to save the xarray in stage 1 folder.

    Args:
        obs (xarray): reduced observations as an xarray.
        output_dir (str): folder where the outputs are saved to.
        filename (str, optional): name to give to the cleaned files.
        Defaults to 'clean_obs'.
    """

    stage1dir = os.path.join(output_dir, 'stage1/')

    if not os.path.exists(stage1dir):
            os.makedirs(stage1dir)

    obs.to_netcdf(os.path.join(stage1dir, f'{filename}.nc'))
