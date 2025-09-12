import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from hustle_tools.read_and_write_config import parse_config
from hustle_tools.read_and_write_config import write_config

from hustle_tools.plotting import plot_one_spectrum
from hustle_tools.plotting import plot_many_spectra
from hustle_tools.plotting import plot_spec_gif
from hustle_tools.plotting import plot_2d_spectra
from hustle_tools.plotting import plot_raw_whitelightcurve
from hustle_tools.plotting import plot_raw_spectrallightcurves
from hustle_tools.plotting import plot_raw_binned_spectrallightcurves
from hustle_tools.plotting import quicklookup
from hustle_tools.plotting import plot_waterfall

from hustle_tools.stage_0 import collect_and_move_files
from hustle_tools.stage_0 import get_files_from_mast
from hustle_tools.stage_0 import locate_target
from hustle_tools.stage_0 import check_subarray

from hustle_tools.stage_1 import load_data_S1
from hustle_tools.stage_1 import save_data_S1
from hustle_tools.stage_1 import correct_hst_flags
from hustle_tools.stage_1 import uniform_value_bkg_subtraction
from hustle_tools.stage_1 import Pagul_bckg_subtraction
from hustle_tools.stage_1 import column_by_column_subtraction
from hustle_tools.stage_1 import track_bkgstars
from hustle_tools.stage_1 import track_0thOrder
from hustle_tools.stage_1 import free_iteration_rejection
from hustle_tools.stage_1 import fixed_iteration_rejection
from hustle_tools.stage_1 import laplacian_edge_detection
from hustle_tools.stage_1 import spatial_smoothing
from hustle_tools.stage_1 import refine_location

from hustle_tools.stage_2 import load_data_S2
from hustle_tools.stage_2 import save_data_S2
from hustle_tools.stage_2 import get_calibration_0th
from hustle_tools.stage_2 import get_trace_solution
from hustle_tools.stage_2 import sens_correct
from hustle_tools.stage_2 import time_and_relative_detrending_in_space
from hustle_tools.stage_2 import determine_ideal_halfwidth
from hustle_tools.stage_2 import standard_extraction
from hustle_tools.stage_2 import optimal_extraction
from hustle_tools.stage_2 import clean_spectra
from hustle_tools.stage_2 import smooth_spectra
from hustle_tools.stage_2 import running_clean_spectra
from hustle_tools.stage_2 import align_spectra
from hustle_tools.stage_2 import align_profiles
from hustle_tools.stage_2 import remove_zeroth_order

from hustle_tools.stage_3 import load_data_S3
from hustle_tools.stage_3 import save_data_S3
from hustle_tools.stage_3 import bin_light_curves
from hustle_tools.stage_3 import clip_light_curves
from hustle_tools.stage_3 import get_state_vectors


def run_pipeline(config_files_dir, stages=(0, 1, 2, 3, 4, 5)):
    """Wrapper for all Stages of the HUSTLE-tools pipeline.

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
            # check for direct image / spec image discrepancies
            xdiscs, ydiscs = check_subarray(os.path.join(stage0_dict['toplevel_dir'],'directimages/or01dr001_flt.fits'),
                                                sorted(glob.glob(os.path.join(os.path.join(stage0_dict['toplevel_dir'],'specimages'),'*flt.fits'))))
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
                        stage0_dict['traces_included'],
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
        
        # supply obs with new attributes if you can
        if stage1_dict['location'] is not None:
            obs.attrs['target_posx'] = stage1_dict['location'][0]
            obs.attrs['target_posy'] = stage1_dict['location'][1]

        # create output directory
        output_dir = os.path.join(stage1_dict['toplevel_dir'],'outputs')
        stage_dir = os.path.join(output_dir,'stage1')
        run_dir = os.path.join(stage_dir,stage1_dict['output_run'])
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # hst flag corrections
        if stage1_dict['do_hst_flags']:
            obs = correct_hst_flags(obs,
                                    stage1_dict['hst_flags'],
                                    stage1_dict['hst_replace'],
                                    verbose=stage1_dict['verbose'],
                                    show_plots=stage1_dict['show_plots'],
                                    save_plots=stage1_dict['save_plots'],
                                    output_dir=run_dir)

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
            refine_location(obs,
                            verbose=stage1_dict['verbose'],
                            show_plots=stage1_dict['show_plots'],
                            save_plots=stage1_dict['save_plots'],
                            output_dir=run_dir)
            # update the dict so that we can see this info in the output config
            stage1_dict['location'] = [obs.attrs['target_posx'],
                                       obs.attrs['target_posy']]

        # displacements by 0th order tracking
        obs["0th_order_pos"] = (("exp_time", "xy"), np.zeros((obs.exp_time.data.shape[0],2))) # placeholder in case you don't do this step
        if stage1_dict['do_0thtracking']:
            # FIX: The below hardcodes an adjustment to your guess that shifts it
            # from direct image pos to spec image. Hardcoding is something that we
            # want to avoid, so we need to think of a better way...
            track_0thOrder(obs, guess=[100,150],
                           verbose=stage1_dict['verbose'],
                           show_plots=stage1_dict['show_plots'],
                           save_plots=stage1_dict['save_plots'],
                           output_dir=run_dir)

        # displacements by background stars
        obs["meanstar_disp"] = (("exp_time", "xy"), np.zeros((obs.exp_time.data.shape[0],2))) # placeholder in case you don't do this step
        if stage1_dict['do_bkg_stars']:
            track_bkgstars(obs, bkg_stars=stage1_dict['bkg_stars_loc'], 
                           window=stage1_dict['bkg_window'],
                           verbose=stage1_dict['verbose'],
                           show_plots=stage1_dict['show_plots'],
                           save_plots=stage1_dict['save_plots'],
                           output_dir=run_dir)
            
        # create quicklook gif
        if stage1_dict['do_quicklook']:
            if stage1_dict['include_hst_dq']:
                obs.data_quality.values += np.where(obs.hst_dq.values >= 1, 1, 0)
                obs.data_quality.values = np.where(obs.data_quality.values >= 1, 1, 0)
            quicklookup(obs,
                        stage1_dict['traces_included'],
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
                                output_dir = run_dir)

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
            
            # tardis clean
            if stage2_dict['do_tardis']:
                obs = time_and_relative_detrending_in_space(obs, trace_x, np.median(trace_y,axis=0),
                                                            wav, order,
                                                            sigma=stage2_dict['tardis_sigma'],
                                                            flux_threshold=stage2_dict['flux_threshold'],
                                                            window=stage2_dict['tardis_window'],
                                                            replacement=stage2_dict['tardis_replace'],
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
                                                    masks=stage2_dict['mask_objs'],
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
                                     sigma=stage2_dict['outlier_sigma'],
                                     verbose=stage2_dict['verbose'])
                
                '''
                spec = smooth_spectra(spec, wav, order=order,
                                      orbit_numbers=obs.orbit_numbers.values,
                                      sigma=stage2_dict['outlier_sigma'],
                                      verbose=stage2_dict['verbose'],
                                      show_plots=stage2_dict['show_plots'], 
                                      save_plots=stage2_dict['save_plots'],
                                      output_dir=run_dir)

                spec = running_clean_spectra(spec, wav, order=order,
                                            sigma=stage2_dict['outlier_sigma'],
                                            kernel_size=5,
                                            verbose=stage2_dict['verbose'],
                                            show_plots=stage2_dict['show_plots'], 
                                            save_plots=stage2_dict['save_plots'],
                                            output_dir=run_dir)
                '''

            # do plotting
            if (stage2_dict['show_plots'] > 0 or stage2_dict['save_plots'] > 0):
                
                plot_one_spectrum(wav, np.median(spec[:, :],axis=0), order,
                                show_plot=(stage2_dict['show_plots'] > 0),
                                save_plot=(stage2_dict['save_plots'] > 0),
                                filename='1Dspec-med_order{}'.format(order),
                                output_dir=run_dir,
                                )
                
                plot_many_spectra(wav, spec, order,
                                  show_plot=(stage2_dict['show_plots'] > 0),
                                save_plot=(stage2_dict['save_plots'] > 0),
                                filename='1Dspec-all_order{}'.format(order),
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
                            os.path.join('outputs/stage2',stage3_dict['input_run']))
        
        # create output directory
        output_dir = os.path.join(stage3_dict['toplevel_dir'],'outputs')
        stage_dir = os.path.join(output_dir,'stage3')
        run_dir = os.path.join(stage_dir,stage3_dict['output_run'])

        for order in stage3_dict['orders']:
            # load the spectrum for this order
            specs = load_data_S3(S3_data_path, order=order)

            # run binning
            light_curves = bin_light_curves(specs, 
                                            order, 
                                            bin_method = stage3_dict['bin_method'],
                                            bins = stage3_dict['wavelength_bins'],
                                            ncol = stage3_dict['N_columns'],
                                            normalize = stage3_dict['normalize'],
                                            norm_lim = len([i for i in specs.orbit_numbers.data if i == 1]), 
                                            rem_exp = None)

            # sigma clip outliers from light curves
            if stage3_dict["sigma_clip"]:
                light_curves = clip_light_curves(light_curves,
                                                 sigma = stage3_dict['sigma_clip'],
                                                 verbose = stage3_dict['verbose'])

            # get jitter vectors 
            state_vectors = get_state_vectors(specs, plot=(stage3_dict['show_plots'] > 0))

            # do plotting
            if (stage3_dict['show_plots'] > 0 or stage3_dict['save_plots'] > 0):

                plot_waterfall(light_curves, 
                               order=order, 
                               show_plot=(stage3_dict['show_plots'] > 0),
                               save_plot=(stage3_dict['save_plots'] > 0),
                               filename=f'waterfall_order{order}',
                               output_dir=run_dir)
                
                plot_raw_binned_spectrallightcurves(light_curves, 
                                                    order=order, 
                                                    show_plot=(stage3_dict['show_plots'] > 0),
                                                    save_plot=(stage3_dict['save_plots'] > 0),
                                                    filename=f'rawslc_order{order}',
                                                    output_dir=run_dir)

            # save light curves
            save_data_S3(light_curves, output_dir=run_dir, order=order)
            
        # write config
        write_config(stage3_dict, stage3_dict['output_run'], 3, run_dir)


            