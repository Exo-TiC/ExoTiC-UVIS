import numpy as np
import matplotlib.pyplot as plt
from exotic_uvis.stage_1 import read_data
from exotic_uvis.stage_1 import corner_bkg_subtraction
from exotic_uvis.stage_1 import track_bkgstars
from exotic_uvis.plotting import plot_exposure
from exotic_uvis.stage_0 import quicklookup
from exotic_uvis.stage_1 import free_iteration_rejection

# data directory of the form '/path to directory with flt files/'
data_dir = '/Users/carlos/Documents/PhD/WASP178/INPUT/'

# output directory of the form '/path to output directory/'
output_dir = '/Users/carlos/Desktop/'

# call quick look up function
quicklookup(data_dir, output_dir)

# read data
#obs = read_data(data_dir, verbose = 0)


'''
# plot one exposure
image = obs.images.data[0]
plot_exposure([image])

# correct cosmic rays
free_iteration_rejection(obs, threshold = 3.5, plot = True, check_all = False)

# try stars displacements
bkg_stars = [[98, 220], 
             [719, 466]]

track_bkgstars(obs,  bkg_stars = bkg_stars, plot = True)



# define corner coordinates
bounds = [[0, 150, 0, 400], [440, 590, 0, 400]]

# try background subtraction
corner_bkg_subtraction(obs, plot = True, bounds = bounds, check_all = False, fit = 'Gaussian')
'''



# Run Stage 2


#config_order_to_parameters(order, config)

#get_calibration_trace()










