import os
import xarray as xr


def save_data(obs, output_dir, filename = 'clean_obs'):

    """
    
    Function to save the xarray in stage 1 folder
    
    """

    stage1dir = os.path.join(output_dir, 'stage1/')

    if not os.path.exists(stage1dir):
            os.makedirs(stage1dir)

    obs.to_netcdf(os.path.join(stage1dir, f'{filename}.nc'))

    