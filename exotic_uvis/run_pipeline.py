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

from exotic_uvis.stage_1 import load_data_S1
from exotic_uvis.stage_1 import save_data_S1
from exotic_uvis.stage_1 import corner_bkg_subtraction
from exotic_uvis.stage_1 import full_frame_bckg_subtraction
from exotic_uvis.stage_1 import Pagul_bckg_subtraction
from exotic_uvis.stage_1 import column_by_column_subtraction
from exotic_uvis.stage_1 import track_bkgstars
from exotic_uvis.stage_1 import track_0thOrder
from exotic_uvis.stage_1 import free_iteration_rejection
from exotic_uvis.stage_1 import fixed_iteration_rejection
from exotic_uvis.stage_1 import laplacian_edge_detection
from exotic_uvis.stage_1 import spatial_smoothing

from exotic_uvis.stage_2 import load_data_S2
from exotic_uvis.stage_2 import save_data_S2
from exotic_uvis.stage_2 import get_trace_solution
from exotic_uvis.stage_2 import determine_ideal_halfwidth
from exotic_uvis.stage_2 import standard_extraction
from exotic_uvis.stage_2 import optimal_extraction
from exotic_uvis.stage_2 import clean_spectra
from exotic_uvis.stage_2 import align_spectra

def run_pipeline(config_files_dir, stages=(0, 1, 2, 3, 4, 5)):
    '''
    Wrapper for all Stages of the ExoTiC-UVIS pipeline.

    :param config_files_dir: str. The path to the folder where all of your ExoTiC-UVIS .hustle files are stored.
    :param stages: tuple of ints from 0 to 5. Which stages you want to execute.
    :return: output products of ExoTiC-UVIS. Locations and details depend on your .hustle files.
    '''
    ######## Run Stage 0 ########
    if 0 in stages:
        # read out the stage 0 config
        stage0_config = glob.glob(os.path.join(config_files_dir,'stage_0*'))[0]
        stage0_dict = parse_config(stage0_config)

        # run data download
        if stage0_dict['do_download']:
            get_files_from_mast(stage0_dict['programID'],
                                stage0_dict['target_name'], 
                                stage0_dict['visit_number'],
                                stage0_dict['toplevel_dir'],
                                extensions=stage0_dict['extensions'])
    
        # collect and move files
        if stage0_dict['do_organize']:
            if not stage0_dict['filesfrom_dir']:
                stage0_dict['filesfrom_dir'] = stage0_dict['toplevel_dir'] # if the data weren't pre-downloaded, then they are here
            collect_and_move_files(stage0_dict['visit_number'], 
                                   stage0_dict['filesfrom_dir'],
                                   stage0_dict['toplevel_dir'])
        
        # locate target in direct image
        if stage0_dict['do_locate']:
            source_x, source_y = locate_target(os.path.join(stage0_dict['toplevel_dir'],'directimages/or01dr001_flt.fits'))
            # modify config keyword
            stage0_dict['location'] = (source_x,source_y)

        # create quicklook gif
        if stage0_dict['do_quicklook']:
            quicklookup(stage0_dict['toplevel_dir'],
                        stage0_dict['gif_dir'], 
                        stage0_dict['verbose'], 
                        stage0_dict['show_plots'], 
                        stage0_dict['save_plots'])

        # write config
        config_dir = os.path.join(stage0_dict['toplevel_dir'],'stage0')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        write_config(stage0_dict, 0, config_dir)


    ####### Run Stage 1 #######
    if 1 in stages:
        # read out the stage 1 config
        stage1_config = glob.glob(os.path.join(config_files_dir,'stage_1*'))[0]
        stage1_dict = parse_config(stage1_config)

        # read the 'location' keyword from the Stage 0 config
        stage0_output_config = os.path.join(stage1_dict['toplevel_dir'],'stage0/stage_0_exoticUVIS.hustle')
        stage0_output_dict = parse_config(stage0_output_config)

        # and grab the location of the source
        stage1_dict['location'] = stage0_output_dict['location']

        # read data
        obs = load_data_S1(stage1_dict['toplevel_dir'], verbose = stage1_dict['verbose'])

        # create output directory
        output_dir = os.path.join(stage1_dict['toplevel_dir'],'outputs')
        run_dir = os.path.join(output_dir,stage1_dict['run_name'])
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)


        # temporal removal fixed iterations
        if stage1_dict['do_fixed_iter']:
            obs = fixed_iteration_rejection(obs,
                                            stage1_dict['fixed_sigmas'],
                                            stage1_dict['replacement'])
        
        # temporal removal free iterations
        if stage1_dict['do_free_iter']:
            obs = free_iteration_rejection(obs,
                                           stage1_dict['free_sigma'], 
                                           verbose = stage1_dict['verbose'],
                                           show_plots = stage1_dict['show_plots'],
                                           save_plots = stage1_dict['save_plots'],
                                           output_dir = run_dir)

        # spatial removal by led
        if stage1_dict['do_led']:
            obs = laplacian_edge_detection(obs, 
                                           sigma = stage1_dict['led_threshold'], 
                                           factor = stage1_dict['led_factor'], 
                                           n = stage1_dict['led_n'], 
                                           build_fine_structure = stage1_dict['fine_structure'], 
                                           contrast_factor = stage1_dict['contrast_factor'])
            
        # spatial removal by smoothing
        if stage1_dict['do_smooth']:
            obs = spatial_smoothing(obs,
                                    sigma=10) # WIP!

        # background subtraction
        if stage1_dict['do_full_frame']:
            obs = full_frame_bckg_subtraction(obs, 
                                              bin_number = stage1_dict['bin_number'], 
                                              fit=stage1_dict['fit'], 
                                              value=stage1_dict['value'])
        
        if stage1_dict['do_corners']:
            obs = corner_bkg_subtraction(obs, bounds=stage1_dict['bounds'],
                                         fit=stage1_dict['value'],
                                         verbose = stage1_dict['verbose'],
                                         show_plots = stage1_dict['show_plots'],
                                         save_plots = stage1_dict['save_plots'],
                                         output_dir=run_dir)
            
        if stage1_dict['do_Pagul23']:
            obs = Pagul_bckg_subtraction(obs,
                                         Pagul_path=stage1_dict['path_to_Pagul23'],
                                         masking_parameter=stage1_dict['mask_parameter'],
                                         median_on_columns=stage1_dict['median_columns'])
            
        if stage1_dict['do_column']:
            obs = column_by_column_subtraction(obs,
                                               rows=stage1_dict['rows'],
                                               sigma=stage1_dict['col_sigma'])

        # displacements by 0th order tracking
        if stage1_dict['do_0thtracking']:
            track_0thOrder(obs,  bkg_stars = stage1_dict['location'])

        # displacements by background stars
        if stage1_dict['do_bkg_stars']:
            track_bkgstars(obs,  bkg_stars = stage1_dict['bkg_stars_loc'], 
                                 verbose_plots=stage1_dict['verbose'],
                                 output_dir=run_dir)

        # save results
        if stage1_dict['do_save']:
            save_data_S1(obs, run_dir)

        # write config
        config_dir = os.path.join(run_dir,'stage1')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        write_config(stage1_dict, 1, config_dir)
        

    ####### Run Stage 2 #######
    if 2 in stages:
        # read out the stage 2 config
        stage2_config = glob.glob(os.path.join(config_files_dir,'stage_2*'))[0]
        stage2_dict = parse_config(stage2_config)

        # read the 'location' keyword from the Stage 0 config
        stage0_output_config = os.path.join(stage2_dict['toplevel_dir'],'stage0/stage_0_exoticUVIS.hustle')
        stage0_output_dict = parse_config(stage0_output_config)

        # and grab the location of the source
        stage2_dict['location'] = stage0_output_dict['location']

        # read data
        obs = load_data_S2(stage2_dict['toplevel_dir'], verbose = stage2_dict['verbose'])

        # create output directory
        output_dir = os.path.join(stage2_dict['toplevel_dir'],'outputs')
        run_dir = os.path.join(output_dir,stage2_dict['run_name'])
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # iterate over orders
        wavs, specs = [], []
        for i, order in enumerate(stage2_dict['traces_to_conf']):
            # configure trace
            trace_x, trace_y, trace_wavs, widths, trace_sens = get_trace_solution(obs,
                                                                      order=order,
                                                                      source_pos=stage2_dict['location'],
                                                                      refine_calibration=stage2_dict['refine_fit'])
            
            # extract
            if stage2_dict['method'] == 'box':
                # determine ideal halfwidth
                if stage2_dict['determine_hw']:
                    halfwidth = determine_ideal_halfwidth(obs,
                                                          trace_x=trace_x,
                                                          trace_y=trace_y,
                                                          wavs=trace_wavs)
                else:
                    halfwidth = stage2_dict['halfwidths_box'][i]
                
                # box extraction
                wav, spec = standard_extraction(obs,
                                                 halfwidth=halfwidth,
                                                 trace_x=trace_x,
                                                 trace_y=trace_y,
                                                 wavs=trace_wavs)
                
            elif stage2_dict['method'] == 'optimum':
                # optimum extraction
                wav, spec = optimal_extraction(obs)

            wavs.append(wav)
            specs.append(spec)

        # align
        if stage2_dict['align']:
            specs, shifts = align_spectra(obs,specs,
                                          trace_x=trace_x,
                                          align=True,
                                          ind1=0,
                                          ind2=-1,
                                          plot_shifts=False)
        
        # clean
        if stage2_dict['outlier_sigma']:
            specs = clean_spectra(specs,
                                  sigma=stage2_dict['outlier_sigma'])
            
        # save 1D spectra
        save_data_S1(obs,outdir='')