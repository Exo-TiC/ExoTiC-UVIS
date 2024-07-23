import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from exotic_uvis.parser import parse_config

from exotic_uvis.stage_0 import quicklookup
from exotic_uvis.stage_0 import collect_and_move_files
from exotic_uvis.stage_0 import get_files_from_mast
from exotic_uvis.stage_0 import locate_target

from exotic_uvis.stage_1 import read_data
from exotic_uvis.stage_1 import corner_bkg_subtraction
from exotic_uvis.stage_1 import full_frame_bckg_subtraction
from exotic_uvis.stage_1 import track_bkgstars
from exotic_uvis.stage_1 import free_iteration_rejection
from exotic_uvis.stage_1 import fixed_iteration_rejection
from exotic_uvis.stage_1 import laplacian_edge_detection




def run_pipeline(config_files_dir):


    ######## Run Stage 0 ########
    stage0_config = glob.glob(os.path.join(config_files_dir,"stage_0*"))[0]
    stage0_dict = parse_config(stage0_config)

    # run data download
    if stage0_dict['do_download']:
        get_files_from_mast(stage0_dict['programID'], stage0_dict['target_name'], 
                            stage0_dict['visit_number'], stage0_dict['MASToutput_dir'], extensions=stage0_dict['extensions'])
   
    # collect and move files
    if stage0_dict['do_organize']:
        collect_and_move_files(stage0_dict['visit_number'], 
                            stage0_dict['filesfrom_dir'], stage0_dict['filesto_dir'])
    
    # locate target in direct image
    if stage0_dict['do_locate']:
        locate_target(stage0_dict['direct_image'])

    # create quicklook gif
    if stage0_dict['do_quicklook']:
        quicklookup(stage0_dict['data_dir'], stage0_dict['gif_dir'])


    ####### Run Stage 1 #######
    stage1_config = glob.glob(os.path.join(config_files_dir,"stage_1*"))[0]
    stage1_dict = parse_config(stage1_config)


    # read data
    obs = read_data(stage1_dict['data_dir'], stage1_dict['output_dir'], verbose = stage1_dict['verbose'])


    # temporal removal fixed iterations
    if stage1_dict['do_fixed_iter']:
        fixed_iteration_rejection(obs, stage1_dict['sigmas'], stage1_dict['replacement'])
    
    # temporal removal free iterations
    if stage1_dict['do_free_iter']:
        free_iteration_rejection(obs, stage1_dict['free_sigma'])

    # spatial removal
    if stage1_dict['do_led']:
        laplacian_edge_detection(obs, 
                                 sigma = stage1_dict['led_threshold'], 
                                 factor = stage1_dict['led_factor'], 
                                 n = stage1_dict['led_n'], 
                                 build_fine_structure = stage1_dict['fine_structure'], 
                                 contrast_factor = stage1_dict['contrast_factor'])


    # background subtraction
    if stage1_dict['do_full_frame']:
        full_frame_bckg_subtraction(obs, 
                                    bin_number = stage1_dict['bin_number'], 
                                    fit=stage1_dict['fit'], 
                                    value=stage1_dict['value'])
    
    if stage1_dict['do_corners']:
        corner_bkg_subtraction(obs, bounds=stage1_dict['bounds'], 
                               fit=stage1_dict['fit'])

    # displacements
    if stage1_dict['do_displacements']:
        track_bkgstars(obs,  bkg_stars = stage1_dict['location'])


    # Run Stage 2


