import numpy as np
from astropy.io import fits
from wfc3tools import sub2full
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import os


def load_data_S2(data_dir, filename = 'clean_obs'):
     
    """

    Function to read in the outputs from stage 1

    """

    obs = xr.open_dataset(os.path.join(data_dir, 'stage1/clean_obs.nc')) 

    return obs




def save_data_S2():



    return 0