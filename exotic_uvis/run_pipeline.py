import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from exotic_uvis.read_and_write_config import parse_config
from exotic_uvis.read_and_write_config import write_config

from exotic_uvis.stage_0 import quicklookup
from exotic_uvis.stage_0 import collect_and_move_files
from exotic_uvis.stage_0 import get_files_from_mast
from exotic_uvis.stage_0 import locate_target

from exotic_uvis.stage_1 import read_data
from exotic_uvis.stage_1 import save_data
from exotic_uvis.stage_1 import corner_bkg_subtraction
from exotic_uvis.stage_1 import full_frame_bckg_subtraction
from exotic_uvis.stage_1 import track_bkgstars
from exotic_uvis.stage_1 import free_iteration_rejection
from exotic_uvis.stage_1 import fixed_iteration_rejection
from exotic_uvis.stage_1 import laplacian_edge_detection




def run_pipeline(config_files_dir, stages=(0, 1, 2, 3, 4, 5)):


    ######## Run Stage 0 ########
    if 0 in stages:
        stage0_config = glob.glob(os.path.join(config_files_dir,"stage_0*"))[0]
        stage0_dict = parse_config(stage0_config)

        # run data download
        if stage0_dict['do_download']:
            get_files_from_mast(stage0_dict['programID'], stage0_dict['target_name'], 
                                stage0_dict['visit_number'], stage0_dict['toplevel_dir'], extensions=stage0_dict['extensions'])
    
        # collect and move files
        if stage0_dict['do_organize']:
            if not stage0_dict['filesfrom_dir']:
                stage0_dict['filesfrom_dir'] = stage0_dict['toplevel_dir'] # if the data weren't pre-downloaded, then they are here
            collect_and_move_files(stage0_dict['visit_number'], 
                                stage0_dict['filesfrom_dir'], stage0_dict['toplevel_dir'])
        
        # locate target in direct image
        if stage0_dict['do_locate']:
            source_x, source_y = locate_target(os.path.join(stage0_dict['toplevel_dir'],"directimages/or01dr001_flt.fits"))
            # modify config keyword
            stage0_dict["location"] = (source_x,source_y)

        # create quicklook gif
        if stage0_dict['do_quicklook']:
            quicklookup(stage0_dict['toplevel_dir'], stage0_dict['gif_dir'])

        # write config
        config_dir = os.path.join(stage0_dict['toplevel_dir'],"stage0")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        write_config(stage0_dict, 0, config_dir)


    ####### Run Stage 1 #######
    if 1 in stages:
        stage1_config = glob.glob(os.path.join(config_files_dir,"stage_1*"))[0]
        stage1_dict = parse_config(stage1_config)

        # read data
        obs = read_data(stage1_dict['toplevel_dir'], verbose = stage1_dict['verbose'])

        # create output directory
        output_dir = os.path.join(stage1_dict['toplevel_dir'],'outputs')
        run_dir = os.path.join(output_dir,stage1_dict['run_name'])
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)


        # temporal removal fixed iterations
        if stage1_dict['do_fixed_iter']:
            fixed_iteration_rejection(obs, stage1_dict['fixed_sigmas'], stage1_dict['replacement'])
        
        # temporal removal free iterations
        if stage1_dict['do_free_iter']:
            free_iteration_rejection(obs, stage1_dict['free_sigma'], 
                                     verbose_plots = stage1_dict['verbose'], output_dir = run_dir)

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

        # displacements by 0th order tracking
        if stage1_dict['do_0thtracking']:
            track_bkgstars(obs,  bkg_stars = stage1_dict['location'])

        # displacements by background stars
        if stage1_dict['do_bkg_stars']:
            track_bkgstars(obs,  bkg_stars = stage1_dict['bkg_stars_loc'], 
                                 verbose_plots=stage1_dict['verbose'],
                                 output_dir = run_dir)

        # save results
        if stage1_dict['do_save']:
            save_data(obs, run_dir)

        # write config
        config_dir = os.path.join(run_dir,"stage1")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        write_config(stage1_dict, 1, config_dir)
        

    ####### Run Stage 2 #######
    if 2 in stages:
        stage2_config = glob.glob(os.path.join(config_files_dir,"stage_2*"))[0]
        stage2_dict = parse_config(stage2_config)