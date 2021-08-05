from shutil import copyfile

import skimage.io as skio
# try:
#     import pygmo
# except ImportError:
#     print("Pygmo is not installed")

import os
import time
import yaml
import numpy as np
import platform

from . import algorithms, objectives, user, utils
from .algorithms import TS_sampler
# from . import simulate_image as simulation

# Define the objectives and regressors here
obj_dict = {"SNR":objectives.Signal_Ratio(75),"Bleach":objectives.Bleach(), "Resolution":objectives.Resolution(pixelsize=None), }
regressors_dict = {"sklearn_BayesRidge":algorithms.sklearn_BayesRidge,
                   "sklearn_GP":algorithms.sklearn_GP,
}


def run_TS(config, save_folder="debug_trial", regressor_name="sklearn_BayesRidge", regressor_args= {"default":{}, "SNR":{}, "bleach":{}}, n_divs_default = 25, param_names = ["p_ex", "p_sted"], with_time=True, default_values_dict={"dwelltime":20e-6},params_conf = { 'p_ex':100e-6, 'p_sted':0, 'dwelltime':10.0e-6,}, x_mins=[400e-6*2**-3, 900e-6*2**-3, ], x_maxs=[400e-6*2**4, 900e-6*2**4], obj_names=["SNR", "bleach"], optim_length = 30, nbre_trials = 2, pareto_option='nsga', borders=None, NSGAII_kwargs=None):
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
    :pareto_option: None if no pareto filtering, 'grid' if grid sort, 'nsga' if NSGA-II
    :borders: None, or List of tuples (minval, maxval) to cap the objective values in the visualization
              for tradeoff selection
    """



    ndims = len(param_names)
    n_points = [n_divs_default]*ndims
    config["computer"] = platform.uname()._asdict()

    # Create directory and save some data
    if not os.path.isfile(save_folder):
        os.mkdir(save_folder)
    root = os.path.dirname(os.path.abspath(__file__))
    copyfile(f"{root}/optim.py", os.path.join(save_folder,"optim.py"))
    copyfile(f"{root}/algorithms.py", os.path.join(save_folder,"algorithms.py"))
    with open(os.path.join(save_folder, "config.yml"), 'w') as f:
        yaml.dump(config, f)
    im_dir_names = ("conf1", "sted", "conf2", "fluomap","X_sample", "y_samples", "pareto_indexes")
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

        s_lb, s_ub, dts, dts_sampling, dts_update = [], [], [], [], []
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
                    col = param_names.index("dwelltime")
                    timesperpixel = X[:,col] #* default_values_dict["pixelsize"] #* (default_values_dict[im_size_nm]*1e-9)**2/default_values_dict[pixelsize]**2w

                # Select a point
                if iter_idx==0:
                    x_selected = X[np.random.randint(X.shape[0])][:, np.newaxis]
                else:
                    if pareto_option == 'grid':
                        # Select a pareto optimal point
                        if with_time:
                            points_arr2d = np.concatenate([y_samples[i] if obj_dict[obj_names[i]].select_optimal==np.argmax else -y_samples[i] for i in range(len(obj_names))]+[-timesperpixel[:,np.newaxis]], axis=1)
                        else:
                            points_arr2d = np.concatenate([y_samples[i] if obj_dict[obj_names[i]].select_optimal==np.argmax else -y_samples[i] for i in range(len(obj_names))], axis=1)
    #                    ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(points=points_arr2d)
                        ndf = utils.pareto_front(points=points_arr2d)
                        np.savetxt(os.path.join(save_folder, "pareto_indexes",str(no_trial), str(iter_idx)+".csv"), ndf, delimiter=",")
                        X_sample = X[ndf,:]
                        y_samples = [y[ndf] for y in y_samples]
                        timesperpixel = timesperpixel[ndf[0]]
                        x_selected = X_sample[user.select(y_samples, [obj_dict[name] for name in obj_names], with_time, timesperpixel, borders=borders), :][:,np.newaxis]
                    else:
                        x_selected = X[user.select(y_samples, [obj_dict[name] for name in obj_names], with_time, timesperpixel, borders=borders), :][:,np.newaxis]
            elif pareto_option == 'nsga':
                # Select a point
                if iter_idx==0:
                    x_selected = np.random.uniform(x_mins, x_maxs)[:, np.newaxis]
                    dt_sampling = 0
                else:
                    sampled_MO_function = algorithms.MO_function_sample(algos, with_time, param_names).evaluate
                    nsga_weigts = [-1 if obj_dict[obj_name]==np.argmin else +1  for obj_name in obj_names]
                    if with_time:
                        nsga_weigts += [-1]
                    print("Calculating the pareto front with NSGA-II...")
                    t0_nsga = time.time()
                    X_sample, logbook, ngens = utils.NSGAII(sampled_MO_function, x_mins, x_maxs, nsga_weigts, **NSGAII_kwargs)
                    dt_sampling = time.time()-t0
                    print(f"The pareto front was calculated in {dt_sampling:.2} seconds with {ngens} generations")
                    y_samples = [algo.sample(X_sample) for algo in algos]
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

            # Acquire conf1, sted_image, conf2
            for i in range(len(x_selected)):
                default_values_dict[param_names[i]] = x_selected[i]
#            p_sted = default_values_dict["p_sted"]
#            default_values_dict["p_sted"] = 0.
            defaults_list = []
            for key in params_conf:
                defaults_list.append(default_values_dict[key])
                default_values_dict[key] = params_conf[key]
            default_values_dict["molecules_disposition"] = None
            sim_data = simulation.simulate_image(**default_values_dict)
            conf1 = sim_data['Acquired signal (photons)']
            skio.imsave(os.path.join(save_folder, "conf1",str(no_trial), str(iter_idx)+".tiff"), conf1)
            default_values_dict["molecules_disposition"] = sim_data["Bleached datamap"]
            skio.imsave(os.path.join(save_folder, "fluomap",str(no_trial), str(iter_idx)+".tiff"), default_values_dict["molecules_disposition"])
#            default_values_dict["p_sted"] = p_sted
            for i, key in enumerate(params_conf):
                default_values_dict[key] = defaults_list[i]
                print("default_values_dict[key]=", default_values_dict[key])
            sim_data = simulation.simulate_image(**default_values_dict)
            sted_image = sim_data['Acquired signal (photons)']
            skio.imsave(os.path.join(save_folder, "sted",str(no_trial), str(iter_idx)+".tiff"), sted_image)
            default_values_dict["molecules_disposition"] = sim_data["Bleached datamap"]
#            default_values_dict["p_sted"] = 0.
            defaults_list = []
            for key in params_conf:
                defaults_list.append(default_values_dict[key])
                default_values_dict[key] = params_conf[key]
            sim_data = simulation.simulate_image(**default_values_dict)
            conf2 = sim_data['Acquired signal (photons)']
            skio.imsave(os.path.join(save_folder, "conf2",str(no_trial), str(iter_idx)+".tiff"), conf2)
            for i, key in enumerate(params_conf):
                default_values_dict[key] = defaults_list[i]

            # foreground on confocal image
            fg_c = utils.get_foreground(conf1)
            # foreground on sted image
            fg_s = utils.get_foreground(sted_image)
            # remove STED foreground points not in confocal foreground, if any
            fg_s *= fg_c

            # Evaluate the objective results
            obj_dict["Resolution"] = objectives.Resolution(pixelsize=default_values_dict["pixelsize"]) #Just in case the pixelsize have changed
            y_result = np.array([obj_dict[name].evaluate([sted_image], conf1, conf2, fg_s, fg_c) for name in obj_names])
            print(x_selected.T, y_result)

#            y_result = obj_func.sample(x_selected.T)
            t0 = time.time()
            [algos[i].update(x_selected.T, y_result[i].flatten()) for i in range(len(obj_names))]
            dt_update = time.time()-t0
            #save s_lb and s_ub, and calculation time
            dts_sampling.append(dt_sampling)
            dts_update.append(dt_update)
#            s_lb.append(algo.s_lb)
#            s_ub.append(algo.s_ub)

            # Save data to text files
            if pareto_option != 'nsga':
                np.savetxt(os.path.join(save_folder,f'X_{no_trial}.csv'), algos[0].X, delimiter=",")
            y_array = np.hstack([algos[i].y[:,np.newaxis] for i in range(len(obj_names))])
            np.savetxt(os.path.join(save_folder,f'y_{no_trial}.csv'), y_array, delimiter=",")
    #        np.savetxt(os.path.join(save_folder,f's_lb_{no_trial}.csv'), np.array(s_lb), delimiter=",")
    #        np.savetxt(os.path.join(save_folder,f's_ub_{no_trial}.csv'), np.array(s_ub), delimiter=",")
            np.savetxt(os.path.join(save_folder,f'dts_sampling_{no_trial}.csv'), np.array(dts_sampling), delimiter=",")
            np.savetxt(os.path.join(save_folder,f'dts_update_{no_trial}.csv'), np.array(dts_update), delimiter=",")
