import os

import numpy as np
import xarray as xr


def load_data_S2(data_dir, filename = 'clean_obs'):
    """Simple function to load the data.

    Args:
        data_dir (str): where the reduced .nc file from Stage 1 is kept.
        filename (str, optional): name of the reduced .nc file.
        Defaults to 'clean_obs'.

    Returns:
        xarray: reduced observations xarray to extract from.
    """

    obs = xr.open_dataset(os.path.join(data_dir, f'stage1/{filename}.nc')) 

    return obs




def save_data_S2(obs, spec, spec_err, 
                 trace_x, trace_y, widths, wavelengths,
                 spec_disp, prof_disp, order = '+1',
                 output_dir = None, filename = 'specs'):
    """Function to create and save xarray containing the information extracted
    from stage 2.

    Args:
        obs (xarray): reduced observations xarray, only needed now for its
        exp_time data.
        specs (list): each extracted spectrum.
        specs_err (list): each extracted spectrum's uncertainties.
        trace_x (list): each extracted spectrum's dispersion solution.
        trace_y (list): each extracted spectrum's spatial solutions.
        widths (list): the width of each extracted trace.
        wavelengths (list): each extracted spectrum's wavelength solution.
        spec_shifts (list): each extracted spectrum's dispersion and
        spatial shifts.
        orders (tuple, optional): which orders are being saved.
        Defaults to ("+1", "-1").
        output_dir (str, optional): where to save the files to.
        Defaults to None.
        filename (str, optional): name to give each file, joined to its order
        string. Defaults to 'specs'.
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
