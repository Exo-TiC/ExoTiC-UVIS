import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from exotic_uvis.read_and_write_config import parse_config
from exotic_uvis.read_and_write_config import write_config

from exotic_uvis.plotting import plot_one_spectrum
from exotic_uvis.plotting import plot_spec_gif
from exotic_uvis.plotting import plot_2d_spectra
from exotic_uvis.plotting import plot_raw_whitelightcurve
from exotic_uvis.plotting import plot_raw_spectrallightcurves
from exotic_uvis.plotting import quicklookup

from exotic_uvis.stage_0 import collect_and_move_files
from exotic_uvis.stage_0 import get_files_from_mast
from exotic_uvis.stage_0 import locate_target

from exotic_uvis.stage_1 import load_data_S1
from exotic_uvis.stage_1 import save_data_S1
from exotic_uvis.stage_1 import uniform_value_bkg_subtraction
from exotic_uvis.stage_1 import Pagul_bckg_subtraction
from exotic_uvis.stage_1 import column_by_column_subtraction
from exotic_uvis.stage_1 import track_bkgstars
from exotic_uvis.stage_1 import track_0thOrder
from exotic_uvis.stage_1 import free_iteration_rejection
from exotic_uvis.stage_1 import fixed_iteration_rejection
from exotic_uvis.stage_1 import laplacian_edge_detection
from exotic_uvis.stage_1 import spatial_smoothing
from exotic_uvis.stage_1 import refine_location

from exotic_uvis.stage_2 import load_data_S2
from exotic_uvis.stage_2 import save_data_S2
from exotic_uvis.stage_2 import get_calibration_0th
from exotic_uvis.stage_2 import get_trace_solution
from exotic_uvis.stage_2 import sens_correct
from exotic_uvis.stage_2 import determine_ideal_halfwidth
from exotic_uvis.stage_2 import standard_extraction
from exotic_uvis.stage_2 import optimal_extraction
from exotic_uvis.stage_2 import clean_spectra
from exotic_uvis.stage_2 import align_spectra
from exotic_uvis.stage_2 import align_profiles
from exotic_uvis.stage_2 import remove_zeroth_order

#from exotic_uvis.stage_3 import load_data_S3
#from exotic_uvis.stage_3 import save_data_S3


def run_pipeline(config_files_dir, stages=(0, 1, 2, 3, 4, 5)):
    """Wrapper for all Stages of the ExoTiC-UVIS pipeline.

    Args:
        config_files_dir (str): folder which contains the .hustle files needed
        to run the stages you want to run.
        stages (tuple, optional): the stages that you want to run.
        Defaults to (0, 1, 2, 3, 4, 5).
    """
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
                                token=stage0_dict['token'],
                                extensions=stage0_dict['extensions'])
    
        # collect and move files
        if stage0_dict['do_organize']:
            if not stage0_dict['filesfrom_dir']:
                stage0_dict['filesfrom_dir'] = stage0_dict['toplevel_dir'] # if the data weren't pre-downloaded, then they are here
            collect_and_move_files(stage0_dict['visit_number'], 
                                   stage0_dict['filesfrom_dir'],
                                   stage0_dict['toplevel_dir'])
            
        # check if output directory exists, otherwise create output directory
        output_dir = os.path.join(stage0_dict['toplevel_dir'],'outputs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # locate target in direct image
        if stage0_dict['do_locate']:
            source_x, source_y = locate_target(os.path.join(stage0_dict['toplevel_dir'],'directimages/or01dr001_flt.fits'))
            # modify config keyword
            stage0_dict['location'] = [source_x,source_y]

        # write config
        config_dir = os.path.join(output_dir,'stage0')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        write_config(stage0_dict, '', 0, config_dir)

        # create quicklook gif
        if stage0_dict['do_quicklook']:
            quicklookup(stage0_dict['toplevel_dir'],
                        stage0_dict['verbose'], 
                        stage0_dict['show_plots'], 
                        stage0_dict['save_plots'],
                        config_dir)


    ####### Run Stage 1 #######
    if 1 in stages:
        # read out the stage 1 config
        stage1_config = glob.glob(os.path.join(config_files_dir,'stage_1*'))[0]
        stage1_dict = parse_config(stage1_config)

        # read the 'location' keyword from the Stage 0 config
        try:
            stage0_output_config = os.path.join(stage1_dict['toplevel_dir'],'outputs/stage0/stage_0_.hustle')
            stage0_output_dict = parse_config(stage0_output_config)

            # and grab the location of the source
            if stage0_output_dict['location'] != None:
                if stage1_dict['verbose'] > 0:
                    print("Stage 0 .hustle located, using 'location' parameter supplied by Stage 0 fitting process.")
                stage1_dict['location'] = stage0_output_dict['location']
        except FileNotFoundError:
            if stage1_dict['verbose'] > 0:
                print("No Stage 0 .hustle located, using 'location' parameter supplied by Stage 1 .hustle instead.")

        # read data
        obs = load_data_S1(stage1_dict['toplevel_dir'],
                           skip_first_fm = stage1_dict['skip_first_fm'],
                           skip_first_or = stage1_dict['skip_first_or'],
                           verbose = stage1_dict['verbose'])

        # create output directory
        output_dir = os.path.join(stage1_dict['toplevel_dir'],'outputs')
        stage_dir = os.path.join(output_dir,'stage1')
        run_dir = os.path.join(stage_dir,stage1_dict['output_run'])
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # temporal removal fixed iterations
        if stage1_dict['do_fixed_iter']:
            obs = fixed_iteration_rejection(obs,
                                            stage1_dict['fixed_sigmas'],
                                            stage1_dict['replacement'], 
                                            verbose=stage1_dict['verbose'],
                                            show_plots=stage1_dict['show_plots'],
                                            save_plots=stage1_dict['save_plots'],
                                            output_dir=run_dir)
        
        # temporal removal free iterations
        if stage1_dict['do_free_iter']:
            obs = free_iteration_rejection(obs,
                                           threshold=stage1_dict['free_sigma'], 
                                           verbose=stage1_dict['verbose'],
                                           show_plots=stage1_dict['show_plots'],
                                           save_plots=stage1_dict['save_plots'],
                                           output_dir=run_dir)

        # spatial removal by led
        if stage1_dict['do_led']:
            obs = laplacian_edge_detection(obs, 
                                           sigma=stage1_dict['led_threshold'], 
                                           factor=stage1_dict['led_factor'], 
                                           n=stage1_dict['led_n'], 
                                           build_fine_structure=stage1_dict['fine_structure'], 
                                           contrast_factor=stage1_dict['contrast_factor'], 
                                           verbose=stage1_dict['verbose'],
                                           show_plots=stage1_dict['show_plots'],
                                           save_plots=stage1_dict['save_plots'],
                                           output_dir=run_dir)
            
        # spatial removal by smoothing
        if stage1_dict['do_smooth']:
            obs = spatial_smoothing(obs,
                                    type=stage1_dict["smth_type"],
                                    kernel=stage1_dict["smth_kernel"],
                                    sigma=stage1_dict["smth_threshold"],
                                    bounds_set=stage1_dict["smth_bounds"],
                                    verbose=stage1_dict['verbose'],
                                    show_plots=stage1_dict['show_plots'],
                                    save_plots=stage1_dict['save_plots'],
                                    output_dir=run_dir) 

        # background subtraction with a uniform bckg value
        if stage1_dict['do_uniform']:
            obs = uniform_value_bkg_subtraction(obs,
                                                fit=stage1_dict['fit'],
                                                bounds=stage1_dict['bounds'],
                                                hist_min=stage1_dict['hist_min'],
                                                hist_max=stage1_dict['hist_max'],
                                                hist_bins=stage1_dict['hist_bins'],
                                                verbose=stage1_dict['verbose'],
                                                show_plots=stage1_dict['show_plots'],
                                                save_plots=stage1_dict['save_plots'],
                                                output_dir=run_dir)
        
        # background subtraction with the Pagul et al. image scaled
        if stage1_dict['do_Pagul']:
            obs = Pagul_bckg_subtraction(obs,
                                         pagul_path=stage1_dict['path_to_Pagul'],
                                         masking_parameter=stage1_dict['mask_parameter'],
                                         smooth_fits=stage1_dict['smooth_fits'],
                                         smooth_parameter=stage1_dict['smoothing_param'],
                                         median_on_columns=stage1_dict['median_columns'],
                                         verbose=stage1_dict['verbose'],
                                         show_plots=stage1_dict['show_plots'],
                                         save_plots=stage1_dict['save_plots'],
                                         output_dir=run_dir)
            
        # background subtraction by columnal medians
        if stage1_dict['do_column']:
            obs = column_by_column_subtraction(obs,
                                               rows=stage1_dict['rows'],
                                               sigma=stage1_dict['col_sigma'],
                                               mask_trace=stage1_dict['mask_trace'],
                                               width=stage1_dict['dist_from_trace'],
                                               verbose=stage1_dict['verbose'],
                                               show_plots=stage1_dict['show_plots'],
                                               save_plots=stage1_dict['save_plots'],
                                               output_dir=run_dir)

        # refine location of the source in the direct image using COM fitting
        if stage1_dict['do_location']:
            refine_location(obs, location=stage1_dict['location'], 
                            verbose=stage1_dict['verbose'],
                            show_plots=stage1_dict['show_plots'],
                            save_plots=stage1_dict['save_plots'],
                            output_dir=run_dir)
            stage1_dict['location'] = [obs.attrs['target_posx'],
                                       obs.attrs['target_posy']]

        # displacements by 0th order tracking
        if stage1_dict['do_0thtracking']:
            # FIX: The below hardcodes an adjustment to your guess that shifts it
            # from direct image pos to spec image. Hardcoding is something that we
            # want to avoid, so we need to think of a better way...
            guess = [stage1_dict['location'][0]+100,
                     stage1_dict['location'][1]+150]
            track_0thOrder(obs, guess=guess,
                           verbose=stage1_dict['verbose'],
                           show_plots=stage1_dict['show_plots'],
                           save_plots=stage1_dict['save_plots'],
                           output_dir=run_dir)

        # displacements by background stars
        obs["meanstar_disp"] = (("exp_time", "xy"), np.zeros((obs.exp_time.data.shape[0],2))) # placeholder in case you don't do this step
        if stage1_dict['do_bkg_stars']:
            track_bkgstars(obs, bkg_stars=stage1_dict['bkg_stars_loc'], 
                           verbose=stage1_dict['verbose'],
                           show_plots=stage1_dict['show_plots'],
                           save_plots=stage1_dict['save_plots'],
                           output_dir=run_dir)
            
        # create quicklook gif
        if stage1_dict['do_quicklook']:
            quicklookup(obs,
                        stage1_dict['verbose'], 
                        stage1_dict['show_plots'], 
                        stage1_dict['save_plots'],
                        run_dir)

        # save results
        if stage1_dict['do_save']:
            save_data_S1(obs, run_dir)

        # write config
        write_config(stage1_dict, stage1_dict['output_run'], 1, run_dir)
        

    ####### Run Stage 2 #######
    if 2 in stages:

        # read out the stage 2 config
        stage2_config = glob.glob(os.path.join(config_files_dir,'stage_2*'))[0]
        stage2_dict = parse_config(stage2_config)

        # read the 'location' keyword from the Stage 0 config
        stage0_output_config = os.path.join(stage2_dict['toplevel_dir'],
                                            'outputs/stage0/stage_0_.hustle')
        stage0_output_dict = parse_config(stage0_output_config)

        # read data
        S2_data_path = os.path.join(stage2_dict['toplevel_dir'],
                                    os.path.join('outputs/stage1',stage2_dict['input_run']))
        obs = load_data_S2(S2_data_path)

        # get the location from the obs.nc file
        stage2_dict['location'] = [obs.attrs['target_posx'], obs.attrs['target_posy']]

        # create output directory
        output_dir = os.path.join(stage2_dict['toplevel_dir'],'outputs')
        stage_dir = os.path.join(output_dir,'stage2')
        run_dir = os.path.join(stage_dir,stage2_dict['output_run'])
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        if stage2_dict['correct_zero']:
            # calibrate 0th order
            x0th, y0th = get_calibration_0th(obs,
                                            source_pos=stage2_dict['location'],
                                            path_to_cal=stage2_dict['path_to_cal'],
                                            verbose=stage2_dict['verbose'],
                                            show_plots=stage2_dict['show_plots'], 
                                            save_plots=stage2_dict['save_plots'],
                                            output_dir=run_dir)
            # test 0th order removal
            remove_zeroth_order(obs, 
                                zero_pos = [x0th, y0th], 
                                rmin = 100, rmax = 300, rwidth = 3, 
                                fit_profile = True,
                                verbose = stage2_dict['verbose'],
                                show_plots = stage2_dict['show_plots'],
                                save_plots = stage2_dict['save_plots'],
                                output_dir = None)

        # iterate over orders
        for i, order in enumerate(stage2_dict['traces_to_conf']):
            # configure trace
            trace_x, trace_y, wav, widths, fs = get_trace_solution(obs,
                                                                   order=order,
                                                                   source_pos=stage2_dict['location'],
                                                                   refine_calibration=stage2_dict['refine_fit'],
                                                                   path_to_cal=stage2_dict['path_to_cal'],
                                                                   verbose=stage2_dict['verbose'],
                                                                   show_plots=stage2_dict['show_plots'], 
                                                                   save_plots=stage2_dict['save_plots'],
                                                                   output_dir=run_dir)
            
            # extract
            if stage2_dict['method'] == 'box':
                # determine ideal halfwidth
                if stage2_dict['determine_hw']:
                    halfwidth = determine_ideal_halfwidth(obs, order,
                                                          trace_x=trace_x,
                                                          trace_y=trace_y,
                                                          wavs=wav,
                                                          indices=stage2_dict['indices'],
                                                          verbose=stage2_dict['verbose'],
                                                          show_plots=stage2_dict['show_plots'], 
                                                          save_plots=stage2_dict['save_plots'],
                                                          output_dir=run_dir)
                    if stage2_dict['verbose'] > 0:
                        print("Optimized half-width for extraction of order {}: {} pixels".format(order,halfwidth))
                else:
                    halfwidth = stage2_dict['halfwidths_box'][i]
                
                # box extraction
                spec, spec_err = standard_extraction(obs,
                                                     halfwidth=halfwidth,
                                                     trace_x=trace_x,
                                                     trace_y=trace_y,
                                                     order=order,
                                                     masks=stage2_dict['mask_objs'],
                                                     verbose=stage2_dict['verbose'],
                                                     show_plots=stage2_dict['show_plots'], 
                                                     save_plots=stage2_dict['save_plots'],
                                                     output_dir=run_dir)
                
            elif stage2_dict['method'] == 'optimal':
                # optimum extraction
                spec, spec_err = optimal_extraction(obs, 
                                                    trace_x, 
                                                    trace_y,
                                                    width = stage2_dict['halfwidths_opt'][i],
                                                    prof_type = stage2_dict['aperture_type'],
                                                    show_plots=stage2_dict['show_plots'],
                                                    save_plots=stage2_dict['save_plots'],
                                                    output_dir=run_dir)

            # do sensitivity
            if stage2_dict['sens_correction']:
                spec, spec_err = sens_correct(spec, spec_err, wav, fs)

            # do alignment
            x_shifts, y_shifts = False, False
            if stage2_dict['align']:
                spec, spec_err, x_shifts = align_spectra(obs,
                                                        spec,
                                                        spec_err,
                                                        order,
                                                        trace_x=wav,
                                                        align=True,
                                                        verbose=stage2_dict['verbose'],
                                                        show_plots=stage2_dict['show_plots'], 
                                                        save_plots=stage2_dict['save_plots'],
                                                        output_dir=run_dir)
                
                y_shifts = align_profiles(obs, 
                                          trace_x,
                                          trace_y,
                                          order,
                                          width = 25,
                                          verbose=stage2_dict['verbose'],
                                          show_plots=stage2_dict['show_plots'], 
                                          save_plots=stage2_dict['save_plots'],
                                          output_dir=run_dir)
                
            # do clean spectra
            if stage2_dict['outlier_sigma']:
                spec = clean_spectra(spec,
                                     sigma=stage2_dict['outlier_sigma'])

            # do plotting
            if (stage2_dict['show_plots'] > 0 or stage2_dict['save_plots'] > 0):
                
                plot_one_spectrum(wav, spec[0, :],
                                show_plot=(stage2_dict['show_plots'] > 0),
                                save_plot=(stage2_dict['save_plots'] > 0),
                                filename='1Dspec_order{}'.format(order),
                                output_dir=run_dir,
                                )
                                
                plot_2d_spectra(wav, spec,
                                show_plot = (stage2_dict['show_plots'] > 0), 
                                save_plot = (stage2_dict['save_plots'] > 0),
                                filename = '2Dspec_order{}'.format(order),
                                output_dir = run_dir)
                
                #plot_raw_spectrallightcurves(obs.exp_time.data, spec, order="+1",
                #                show_plot = (stage2_dict['show_plots'] > 0), 
                #                save_plot = (stage2_dict['save_plots'] > 0),
                #                filename = 's2_2Dspec_order{}'.format(order),
                #                output_dir = run_dir)
                
                plot_raw_whitelightcurve(obs.exp_time.data,spec,
                                         order=order,
                                         show_plot = (stage2_dict['show_plots'] > 0), 
                                         save_plot = (stage2_dict['save_plots'] > 0),
                                         filename = 'rawwlc_order{}'.format(order),
                                         output_dir = run_dir)

                plot_spec_gif(wav,spec,
                              order=order,
                              show_plot=(stage2_dict['show_plots'] > 0),
                              save_plot=(stage2_dict['save_plots'] > 0),
                              filename='1Dspec_order{}'.format(order),
                              output_dir=run_dir)
           
            # save order xarray
            save_data_S2(obs, 
                         spec, 
                         spec_err, 
                         trace_x, 
                         trace_y, 
                         widths, 
                         wav,
                         x_shifts, 
                         y_shifts,
                         order, 
                         output_dir=run_dir)


        # write config
        write_config(stage2_dict, stage2_dict['output_run'], 2, run_dir)
        

    ####### Run Stage 3 #######
    if 3 in stages:
        # read out the stage 3 config
        stage3_config = glob.glob(os.path.join(config_files_dir,'stage_3*'))[0]
        stage3_dict = parse_config(stage3_config)

        # read data, one order at a time
        S3_data_path = os.path.join(stage3_dict['toplevel_dir'],
                                    os.path.join('outputs',stage3_dict['input_run']))

        #for order in stage3_dict['orders']:
            # load the spectrum for this order
            #spex = load_data_S3(S3_data_path, order=order, verbose = stage3_dict['verbose'])

            