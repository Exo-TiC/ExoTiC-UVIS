import numpy as np
from astropy.io import fits
from wfc3tools import sub2full
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import os


def load_data_S2(data_dir, filename = 'clean_obs', verbose = 0):
     
    """

    Function to read in the outputs from stage 1

    """

    obs = xr.open_dataset(os.path.join(data_dir, 'stage1/clean_obs.nc')) 

    return obs




def save_data_S2(obs, spec, spec_err, 
                 trace_x, trace_y, widths, wavelengths,
                 spec_disp, prof_disp, order = '+1',
                 output_dir = None, filename = 'specs'):
    """Function to create and save xarray containing the information extracted
    from stage 2.

    Args:
        obs (_type_): _description_
        specs (_type_): _description_
        specs_err (_type_): _description_
        trace_x (_type_): _description_
        trace_y (_type_): _description_
        widths (_type_): _description_
        wavelengths (_type_): _description_
        spec_shifts (_type_): _description_
        orders (tuple, optional): _description_. Defaults to ("+1", "-1").
        output_dir (_type_, optional): _description_. Defaults to None.
        filename (str, optional): _description_. Defaults to 'specs'.

    Returns:
        _type_: _description_
    """

    # Create and save xarray for each order
    spectra = xr.Dataset(
        data_vars=dict(
            spec = (['exp_time', 'x'], spec),
            spec_err = (['exp_time', 'x'], spec_err),
            trace = (['exp_time', 'x'], trace_y),
            ),
        coords=dict(
            wave=(['x'], wavelengths),
            trace_x=(['x'], trace_x),
            exp_time = obs.exp_time.data,
            ),
        )
    
    if spec_disp is not False:
        spectra['spec_disp'] = xr.DataArray(spec_disp, dims=['exp_time']) 
    if prof_disp is not False:
        spectra['prof_disp'] = xr.DataArray(prof_disp, dims=['exp_time', 'x']) 
    if widths is not False:
        spectra['fit_widths'] = xr.DataArray(widths, dims=['exp_time', 'x']) 
        
    #for i in range(len(bkg_stars)):
    #    spectra['stars{}_disp'.format(i + 1)] = obs['star{}_disp'.format(i)]   
    spectra['meanstar_disp'] = obs['meanstar_disp']

    # Save results in Stage 3 folder 
    stage2dir = os.path.join(output_dir, 'stage2/')

    if not os.path.exists(stage2dir):
            os.makedirs(stage2dir)

    spectra.to_netcdf(os.path.join(stage2dir, f'{filename}_{order}.nc'))

    return 0