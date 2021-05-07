from shutil import copyfile
from sklearn.preprocessing import PolynomialFeatures
#import algorithms
from .algorithms import TS_sampler, regressor_name_dict
import os
import time
import yaml
import pickle
import sys
import numpy as np
import platform
#from .analysis import poly_without_interactions


#class poly_without_interactions():
#    """Represents a multivariate polynomial without interactions between the features
#
#    Parameters
#    ----------
#    coeffs_list_of_arrays : list of column vectors (numpy arrays)
#        The list of coefficient arrays for each variable.
#    noise_std : float
#        Standard deviation of the gaussian noise associated with the polynomial
#    Attributes
#    ----------
#    degree_ : int
#        Degree of the polynomial
#    n_params_ : int
#        Number of parameters of the polynomial
#    coeffs_ : numpy column array
#        Coefficients of the polynomial. In the order a1, b1x1, c1x1^2, ... ,a2, b2x2, c2x2^2, ...
#    """
#
#    def __init__(self, coeffs_list_of_list, noise_std):
#        self.coeffs_list_of_list = coeffs_list_of_list
#        self.noise_std = noise_std
#        self.degree_ = len(coeffs_list_of_list[0]) - 1
#        self.n_params_ = len(coeffs_list_of_list)
##         self.coeffs_ = np.concatenate(list(coeffs_dict.values())[:ndims])
#        self.coeffs_ = np.concatenate(coeffs_list_of_list)
#
#    def evaluate(self, X):
#        poly = PolynomialFeatures(self.degree_)
#        poly_features = np.hstack([poly.fit_transform(X[:,i][:,np.newaxis]) for i in range(self.n_params_)])
#        return poly_features@self.coeffs_
#    def sample(self, X):
#        return self.evaluate(X) + np.random.normal(0,self.noise_std, X.shape[0])



## def kernel_TS(obj_coeffs_file, bw_factor):
#if __name__ == "__main__":
##    obj_coeffs_file = sys.argv[1]
#    obj_pkl_file = sys.argv[1]
#    save_folder = sys.argv[2]
##    bw_factor = float(sys.argv[3])
#    kernel_str = sys.argv[3]
#    single_param_obj = sys.argv[4] #True/False

def run_TS( save_folder, regressor_name, regressor_args, n_points_default = 25, obj_pkl_file="poly4d_FWHM_BLEACH.pkl",optim_length = 200, nbre_trials = 10, single_param_obj=False):

#    # This completely defines the objective function
#    with open(obj_coeffs_file, 'r') as f:
#        obj_coeffs_list_of_list = yaml.load(f, Loader=yaml.FullLoader)["obj_coeffs"]
#    obj_coeffs_list_of_list = [np.array(x)[:,np.newaxis] for x in obj_coeffs_list_of_list]
    


#    obj_noise_std = 0.1

    # Optimizations parameters
    import pickle
    with open(obj_pkl_file, 'rb') as f:
        obj_func = pickle.load(f)
    if single_param_obj:
        ndims = 1
    else:
        ndims = int(obj_func.ndims_)
    x_min_default = -1
    x_max_default = 1
    x_mins   = [x_min_default]*ndims
    x_maxs   = [x_max_default]*ndims
#    n_points_default = 50
#    if kernel_str:
#        n_points_default = 7
#    else:
#        n_points_default = 25
    n_points = [n_points_default]*ndims


#    optim_length = 200

    # Regression parameters
#    bandwidth = (x_max_default - x_min_default)*ndims/3 * bw_factor
#    noise_lb = 0.1
#    noise_ub = 0.1
#    noise_level = noise_level

    # Experiment parameters
#    nbre_trials = 30
    save_folder = save_folder


    


    config = {
        #Computer characteristics
        "computer": platform.uname()._asdict(),

        # This completely defines the objective function
#        "obj_coeffs_list_of_list" : [x.flatten().tolist() for x in obj_coeffs_list_of_list],
#        "obj_noise_std" : obj_noise_std,
        "obj_pkl_file":obj_pkl_file,

        # Optimizations parameters
        "ndims": ndims,
        "x_min_default": x_min_default,
        "x_max_default" : x_max_default,
        "x_mins"   : x_mins,
        "x_maxs"   : x_maxs,
        "n_points_default" : n_points_default,
        "n_points" : n_points,

        "regressor_name": regressor_name,

        "optim_length" : optim_length,

        # Regression parameters
#        "bandwidth" : bandwidth,
#        "noise_lb" : noise_lb,
#        "noise_ub" : noise_ub,
#        "kernel_str":kernel_str,
#        "alpha":alpha,
#        "poly_deg":poly_deg,
#        "poly_noise":poly_noise,
#        "poly_reg":poly_reg,
        "regressor_args":regressor_args,
        

        # Experiment parameters
        "nbre_trials" : nbre_trials,
        "save_folder" : save_folder,

    }
    config["username"] = config["computer"]['system']
    
    if not os.path.isfile(save_folder):
        os.mkdir(save_folder)
    copyfile("src/TS_fitted_poly.py", os.path.join(save_folder,"TS_fitted_poly.py"))
    copyfile("src/algorithms.py", os.path.join(save_folder,"algorithms.py"))
    copyfile(obj_pkl_file, os.path.join(save_folder,obj_pkl_file))
    with open(os.path.join(save_folder, "config.yml"), 'w') as f:
        yaml.dump(config, f)
    
    np.random.seed(None)
    rand_state = np.random.get_state()
    with open(os.path.join(save_folder, "ran_state.pkl"), 'wb') as f:
        pickle.dump(rand_state, f)

#    algo_names_dict = {
#        "Kernel_TS" : algorithms.Kernel_TS,
#        "Kernel_TS_cholesky" : algorithms.Kernel_TS_cholesky,
#        "Kernel_TS_eigh" : algorithms.Kernel_TS_eigh,
#        "custom_GP_TS" : algorithms.custom_GP_TS, #"Basic" sklearn rgp regression class used for regression
#        "poly_reg_TS" : algorithms.poly_reg_TS,
#    }

    


#    obj_func = poly_without_interactions(obj_coeffs_list_of_list, obj_noise_std)

    

    for no_trial in range(nbre_trials):
        print(f"Trial {no_trial}...")
        #Define the algo
#        algo = algo_names_dict[algo_name](bandwidth, noise_lb, noise_ub)
#        if kernel_str:
#            algo = algo_names_dict[algo_name](kernel_str, alpha)
#        elif poly_deg:
#            algo = algo_names_dict[algo_name](algorithms.linear_regression(poly_noise, poly_reg), poly_deg)
        algo = TS_sampler(regressor_name_dict[regressor_name](**regressor_args))
        # Define the parameter values to predict
        grids = np.meshgrid(*[np.linspace(x_mins[i], x_maxs[i], n_points[i]) for i in range(ndims)])
        X = np.hstack([grid.ravel()[:,np.newaxis] for grid in grids]) 

        s_lb, s_ub, dts, dts_sampling, dts_update = [], [], [], [], []
        for i in range(optim_length):
#             print(f"    iter {i}...")
            # Sample, select the parameter which argmax 
            t0 = time.time()
            y_samples = algo.sample(X)
            dt_sampling = time.time()-t0
            x_selected = X[np.argmax(y_samples), :][:, np.newaxis]
            # Try the action and update the posterior
            if ndims == 1:
                y_result = obj_func.sample_1D(x_selected.T)
            else:
                y_result = obj_func.sample(x_selected.T)
            t0 = time.time()
            algo.update(x_selected.T, y_result.flatten())
            dt_update = time.time()-t0
            #save s_lb and s_ub, and calculation time
            dts_sampling.append(dt_sampling)
            dts_update.append(dt_update)
#            s_lb.append(algo.s_lb)
#            s_ub.append(algo.s_ub)

        # Save the data
        np.savetxt(os.path.join(save_folder,f'X_{no_trial}.csv'), algo.X, delimiter=",")
        np.savetxt(os.path.join(save_folder,f'y_{no_trial}.csv'), algo.y, delimiter=",")
#        np.savetxt(os.path.join(save_folder,f's_lb_{no_trial}.csv'), np.array(s_lb), delimiter=",")
#        np.savetxt(os.path.join(save_folder,f's_ub_{no_trial}.csv'), np.array(s_ub), delimiter=",")
        np.savetxt(os.path.join(save_folder,f'dts_sampling_{no_trial}.csv'), np.array(dts_sampling), delimiter=",")
        np.savetxt(os.path.join(save_folder,f'dts_update_{no_trial}.csv'), np.array(dts_update), delimiter=",")
