# TESTED ON THE ENVIRONMENT py3kclone2

from shutil import copyfile
from . import algorithms, objectives, user, utils, microscope, split_acquire

from .algorithms import TS_sampler

# import src.simulate_image as simulation
import skimage.io as skio
# import pygmo

import os
import time
import yaml
import numpy as np
import platform

import pickle
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


from inspect import currentframe, getframeinfo
import functools



config_conf = microscope.get_config("Setting confocal configuration.", "conf_logo.png")
config_sted = microscope.get_config("Setting STED configuration.", "sted_logo.png")



params_set = {"dwelltime": microscope.set_dwelltime,
              "p_ex": functools.partial(microscope.set_power, laser_id=5),
              "p_sted": functools.partial(microscope.set_power, laser_id=6),
              "line_step": functools.partial(microscope.set_linestep, step_id=0),
              "pixelsize": functools.partial(microscope.set_pixelsize),
              # "Rescue/Signal_Level": functools.partial(microscope.set_rescue_signal_level, channel_id=0),
              # "Rescue/Strength": functools.partial(microscope.set_rescue_strength, channel_id=0)
              }











# Define the objectives and regressors here
obj_dict = {"SNR":objectives.Signal_Ratio(75),"Bleach":objectives.Bleach(), "Resolution":objectives.Resolution(pixelsize=None), }
regressors_dict = {"sklearn_BayesRidge":algorithms.sklearn_BayesRidge,
                   "sklearn_GP":algorithms.sklearn_GP,
}

def run_TS(config, save_folder="debug_trial", regressor_name="sklearn_BayesRidge", regressor_args= {"default":{}, "SNR":{}, "bleach":{}}, n_divs_default = 25, n_points=None, param_names = ["p_ex", "p_sted"], with_time=True, default_values_dict={"dwelltime":20e-6},params_conf = { 'p_ex':100e-6, 'p_sted':0, 'dwelltime':10.0e-6,}, x_mins=[400e-6*2**-3, 900e-6*2**-3, ], x_maxs=[400e-6*2**4, 900e-6*2**4], obj_names=["SNR", "bleach"], optim_length = 30, nbre_trials = 2, pareto_only=False, borders=None, split_sted_params=None, time_limit=None, pareto_option='nsga', NSGAII_kwargs=None):
    """This function does multi-objective Thompson sampling optimization of parameters of simulated STED images.

    :param config: Dictionary of all the function parameters to be saved as a yaml file
    :save_folder: Directory that will be created to store the optimization data
    :regressor_name: Name (str key of a dictionary) of the regressor class
    :regressor_args: Dictionary of the arguments of the regressor class arguments
    :n_divs_default: Number (int) of discretizations of the paramter space along each axis
    :param_names: Parameters (str name of arguments for the STED simulator) to optimize
    :with_time: True if the imaging time is optimized
    :default_values_dict: Default argument values dictionary for the STED simulator (STED)
    :params_conf: Default argument values dictionary for the STED simulator (confocals 1 and 2)
    :x_mins: List of the smallest parameter values of the parameter space (same order as param_names)
    :x_maxs: List of the largest parameter values of the parameter space (same order as param_names)
    :obj_names: Name (str key of a dictionary) of objectives
    :optim_length: Number of iterations of an optimization
    :nbre_trials: Number of trials
    :borders: None, or List of tuples (minval, maxval) to cap the objective values in the visualization
              for tradeoff selection
    """


    # Set the default sted parameters
    for name, value in default_values_dict.items():
        params_set[name](config_sted, float(value))



    ndims = len(param_names)

    if n_points is None:
        n_points = [n_divs_default]*ndims

    config["computer"] = platform.uname()._asdict()


    # Create directory and save some data
    if not os.path.isfile(save_folder):
        os.mkdir(save_folder)
    copyfile("banditopt/optim.py", os.path.join(save_folder,"optim.py"))
    copyfile("banditopt/algorithms.py", os.path.join(save_folder,"algorithms.py"))

    copyfile('run_optim_example_microscope.py', os.path.join(save_folder,"run_optim_example_microscope.py"))
    if split_sted_params is not None:
        copyfile('run_optim_example_split-sted.py', os.path.join(save_folder,"run_optim_example_split-sted.py"))

    with open(os.path.join(save_folder, "config.yml"), 'w') as f:
        yaml.dump(config, f)

    with open(os.path.join(save_folder, "imspector_config_confocal"), "w") as f:
        config = config_conf.parameters("")
        yaml.dump(config, f)
    with open(os.path.join(save_folder, "imspector_config_sted"), "w") as f:
        config = config_sted.parameters("")
        yaml.dump(config, f)

    im_dir_names = ["conf1", "sted", "conf2", "fluomap","X_sample",  "y_samples", "pareto_indexes"]
    if "pixelsize" in param_names:
        im_dir_names += ["conf0"]
        config_conf_varpixel = microscope.get_config("Setting confocal configuration.", "conf_varpixel_logo.png")
    if split_sted_params is not None:
        im_dir_names.append("sted_stack")
    for dir_name in im_dir_names:
        os.mkdir(os.path.join(save_folder, dir_name))


    for no_trial in range(nbre_trials):
        print(f"Trial {no_trial}...")
        for dir_name in im_dir_names:
            os.mkdir(os.path.join(save_folder, dir_name,str(no_trial)))
        #Define the algos
        algos = []
        for name in obj_names:
            args = regressor_args["default"].copy()
            for key, value in regressor_args[name].items():
                args[key] = value
            algos.append(TS_sampler(regressors_dict[regressor_name](**args)))
        # Define the parameter values to predict
        if pareto_option != 'nsga':
            grids = np.meshgrid(*[np.linspace(x_mins[i], x_maxs[i], n_points[i]) for i in range(ndims)])
            X = np.hstack([grid.ravel()[:,np.newaxis] for grid in grids])


        # regions = []
        # while len(regions) < optim_length:
        #     # regions.append(user.get_regions(overview_name='640 {0}'))
        #     print('regions=', regions)
        #     input("Select a new region on the microscope, then press Enter to continue...")
        #     regions += user.get_regions(config=config_overview, overview_name='640 {0}')

        config_overview = microscope.get_config("Setting STED configuration.", 'overview_logo.png')
        import pdb; pdb.set_trace()
        regions = user.get_regions(config=config_overview, overview_name='640 {0}')
        regions.reverse()

        s_lb, s_ub, dts, dts_sampling, dts_update, dts_sted = [], [], [], [], [], []
        for iter_idx in range(optim_length):
#             print(f"    iter {i}...")
            # Sample objective values over the parameter space
            t0 = time.time()

            if pareto_option != 'nsga':
                y_samples = [algo.sample(X) for algo in algos]
                np.savetxt(os.path.join(save_folder, "y_samples",str(no_trial), str(iter_idx)+".csv"), np.dstack(y_samples).squeeze(), delimiter=",")



                dt_sampling = time.time()-t0
                if "dwelltime" not in param_names:
                    timesperpixel = np.ones((X.shape[0], 1)) * default_values_dict["dwelltime"] #* default_values_dict["pixelsize"] * (default_values_dict[im_size_nm]*1e-9)**2/default_values_dict[pixelsize]**2
                    timesperpixel = timesperpixel.flatten()
                else:
                    #TODO: I should correct those condition for more generality
                    if 'line_step' in param_names:
                        timesperpixel = X[:,param_names.index("dwelltime")]*X[:,param_names.index("line_step")]*X[:,param_names.index("pixelsize")]**2/(20e-9)**2
                    else:
                        timesperpixel = X[:,param_names.index("dwelltime")]


                # Select a point
                ndt = np.arange(X.shape[0])
                ndf = np.arange(X.shape[0])
                if time_limit is not None:
                    ndt = np.arange(X.shape[0])[timesperpixel<=time_limit]
                    y_samples = [y[ndt,:] for y in y_samples]
                # if  iter_idx < 4:
                if  iter_idx == 0:
                    X_sample = X[ndt,:]
                    x_selected = X_sample[np.random.randint(X_sample.shape[0])][:, np.newaxis]
                else:
                    if pareto_only:
                        # Select a pareto optimal point
                        if with_time:
                            points_arr2d = np.concatenate([y_samples[i] if obj_dict[obj_names[i]].select_optimal==np.argmin else -y_samples[i] for i in range(len(obj_names))]+[timesperpixel[:,np.newaxis]], axis=1)
                        else:
                            points_arr2d = np.concatenate([y_samples[i] if obj_dict[obj_names[i]].select_optimal==np.argmin else -y_samples[i] for i in range(len(obj_names))], axis=1)
                        print('Calculating pareto front...')
                        # ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(points=points_arr2d)
                        ndf = np.array(utils.pareto_front(points=points_arr2d))
                        print('DONE Calculating pareto front')
                    np.savetxt(os.path.join(save_folder, "pareto_indexes",str(no_trial), str(iter_idx)+".csv"), ndt[ndf], delimiter=",")
                    X_sample = X[ndt,:][ndf,:]
                    y_samples = [y[ndf] for y in y_samples]
                    timesperpixel = timesperpixel[ndf]
    #                    x_selected = X_sample[user.select(y_samples, [obj_dict[name] for name in obj_names], with_time, timesperpixel, borders=borders), :][:,np.newaxis]




                    x_selected = X_sample[user.select(y_samples, [obj_dict[name] for name in obj_names], with_time, timesperpixel, borders=borders), :][:,np.newaxis]
            elif pareto_option == 'nsga':
                # Select a point
                if iter_idx==0:
                    time_pred=np.inf
                    while time_pred>time_limit:
                        x_selected = np.random.uniform(x_mins, x_maxs)[:, np.newaxis]
                        time_pred = x_selected[param_names.index("dwelltime")]*x_selected[param_names.index("line_step")] * x_selected[param_names.index("pixelsize")]**2/(20e-9)**2
                        print(time_pred, time_limit, time_pred>time_limit)
                    dt_sampling = 0
                else:
                    sampled_MO_function_obj = algorithms.MO_function_sample(algos, with_time, param_names, time_limit=time_limit, borders=borders)
                    sampled_MO_function = sampled_MO_function_obj.evaluate
                    nsga_weigts = [-1 if obj_dict[obj_name].select_optimal==np.argmin else +1  for obj_name in obj_names]
                    if with_time:
                        nsga_weigts += [-1]
                    print("Calculating the pareto front with NSGA-II...")
                    t0_nsga = time.time()
                    X_sample, logbook, ngens = utils.NSGAII(sampled_MO_function, x_mins, x_maxs, nsga_weigts, **NSGAII_kwargs)
                    dt_sampling = time.time()-t0
                    print(f"The pareto front was calculated in {dt_sampling:.2} seconds with {ngens} generations")



                    if time_limit is not None:
                        if 'line_step' in param_names:
                            timesperpixel = X_sample[:,param_names.index("dwelltime")]*X_sample[:,param_names.index("line_step")]*X_sample[:,param_names.index("pixelsize")]**2/(20e-9)**2
                        else:
                            timesperpixel = X_sample[:,param_names.index("dwelltime")]
                        X_sample = X_sample[timesperpixel<=time_limit]

                    y_samples = [algo.sample(X_sample, seed=sampled_MO_function_obj.seeds[i_seed]) for i_seed, algo in enumerate(algos)]

                    if with_time:
                        dwelltime_pos = list(X[0,:].flatten()).index('dwelltime')
                        timesperpixel = X_sample[:, dwelltime_pos]
                    else:
                        timesperpixel = np.ones((X_sample.shape[0], 1))
                    x_selected = X_sample[user.select(y_samples, [obj_dict[name] for name in obj_names], with_time, timesperpixel, borders=borders), :][:,np.newaxis]

                    np.savetxt(os.path.join(save_folder, "X_sample",str(no_trial), str(iter_idx)+".csv"), X_sample, delimiter=",")
                    np.savetxt(os.path.join(save_folder, "y_samples",str(no_trial), str(iter_idx)+".csv"), np.dstack(y_samples).squeeze(), delimiter=",")

            else:
                raise ValueError(f"The pareto_option {pareto_option} does not exists")





            print("x_selected=", x_selected)




            if len(regions)==0:
                print("Now is the time to select a new overview region.")
                os.system('pause')
                config_overview = microscope.get_config("Setting STED configuration.", 'overview_logo.png')
                regions = user.get_regions(config=config_overview, overview_name='640 {0}')
                regions.reverse()
                print('print(regions)', regions)
            # x, y = regions[iter_idx]
            print('print(regions)', regions)
            x, y = regions.pop()
            print('print(x, y)',x, y)
            print('print(regions)', regions)
            microscope.set_offsets(config_conf, x, y)
            if "pixelsize" in param_names:
                microscope.set_offsets(config_conf_varpixel, x, y)
            microscope.set_offsets(config_sted, x, y)
            # Acquire conf1, sted_image, conf2

            if "pixelsize" in param_names:
                params_set["pixelsize"](config_conf_varpixel, float(x_selected[param_names.index('pixelsize')]))
                stacks, _ = microscope.acquire(config_conf_varpixel)
                conf0 = stacks[0][0]
                skio.imsave(os.path.join(save_folder, "conf0",str(no_trial), str(iter_idx)+".tiff"), conf0, check_contrast=False)
            stacks, _ = microscope.acquire(config_conf)
            conf1 = stacks[0][0]
            skio.imsave(os.path.join(save_folder, "conf1",str(no_trial), str(iter_idx)+".tiff"), conf1, check_contrast=False)
            if split_sted_params is not None:
                t0_sted = time.time()
                split_sted_params["config_conf"] = config_conf
                split_sted_params["config_sted"] = config_sted
                for i in range(len(x_selected)):
                    # import pdb; pdb.set_trace()
                    split_sted_params[param_names[i]] = float(x_selected[i])
                sted_image, sted_stack = split_acquire.acquire(**split_sted_params)
                sted_image = np.nan_to_num(sted_image)

                dt_sted = time.time()-t0_sted

                skio.imsave(os.path.join(save_folder, "sted_stack",str(no_trial), str(iter_idx)+".tiff"), sted_stack, check_contrast=False)
            else:
                for i in range(len(x_selected)):
                    params_set[param_names[i]](config_sted, float(x_selected[i]))
                    print(f'------------------ Parameter {param_names[i]} set to {float(x_selected[i])}-----------')
                stacks, _ = microscope.acquire(config_sted)
                t0_sted = time.time()
                sted_image = stacks[0][0]

                dt_sted = time.time()-t0_sted

            skio.imsave(os.path.join(save_folder, "sted",str(no_trial), str(iter_idx)+".tiff"), sted_image.astype(np.float32), check_contrast=False)
            stacks, _ = microscope.acquire(config_conf)
            conf2 = stacks[0][0]
            skio.imsave(os.path.join(save_folder, "conf2",str(no_trial), str(iter_idx)+".tiff"), conf2, check_contrast=False)








            # foreground on confocal image
            fg_c = utils.get_foreground(conf1)
            # foreground on sted image
            if np.any(sted_image):
                fg_s = utils.get_foreground(sted_image)
            else:
                fg_s = np.ones_like(sted_image)
            # remove STED foreground points not in confocal foreground, if any
            if "pixelsize" in param_names:
                fg_s *= utils.get_foreground(conf0)
            else:
                fg_s *= fg_c

            # Evaluate the objective results
            if "Resolution" in obj_names:
                if "pixelsize" in param_names:
                    obj_dict["Resolution"] = objectives.Resolution(pixelsize=x_selected[param_names.index('pixelsize')], res_cap=borders[obj_names.index("Resolution")][1]) #Just in case the pixelsize have changed
                else:
                    obj_dict["Resolution"] = objectives.Resolution(pixelsize=default_values_dict["pixelsize"], res_cap=borders[obj_names.index("Resolution")][1]) #Just in case the pixelsize have changed
            y_result = np.array([obj_dict[name].evaluate([sted_image], conf1, conf2, fg_s, fg_c) for name in obj_names])
            print(x_selected.T, y_result)

#            y_result = obj_func.sample(x_selected.T)
            t0 = time.time()
            [algos[i].update(x_selected.T, y_result[i].flatten()) for i in range(len(obj_names))]
            dt_update = time.time()-t0
            #save s_lb and s_ub, and calculation time
            dts_sampling.append(dt_sampling)
            dts_update.append(dt_update)
            dts_sted.append(dt_sted)
#            s_lb.append(algo.s_lb)
#            s_ub.append(algo.s_ub)

            # Save data to text files

            np.savetxt(os.path.join(save_folder,f'X_{no_trial}.csv'), algos[0].X, delimiter=",")
            y_array = np.hstack([algos[i].y[:,np.newaxis] for i in range(len(obj_names))])
            np.savetxt(os.path.join(save_folder,f'y_{no_trial}.csv'), y_array, delimiter=",")
    #        np.savetxt(os.path.join(save_folder,f's_lb_{no_trial}.csv'), np.array(s_lb), delimiter=",")
    #        np.savetxt(os.path.join(save_folder,f's_ub_{no_trial}.csv'), np.array(s_ub), delimiter=",")
            np.savetxt(os.path.join(save_folder,f'dts_sampling_{no_trial}.csv'), np.array(dts_sampling), delimiter=",")
            np.savetxt(os.path.join(save_folder,f'dts_update_{no_trial}.csv'), np.array(dts_update), delimiter=",")
            np.savetxt(os.path.join(save_folder,f'dts_sted_{no_trial}.csv'), np.array(dts_update), delimiter=",")

#            print("line =", getframeinfo(currentframe()).lineno, "iter_idx=", iter_idx,"np.unique(X[:,0]) =", np.unique(X[:,0]))
