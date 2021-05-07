import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import pickle
#TODO: replace src. by the "correct" import method (in other src file too, . or src. add __init__.py?)
import src.algorithms as algorithms
from src.algorithms import regressor_name_dict, TS_sampler, sklearn_BayesRidge
#from src.algorithms import obj1D_dict as obj_dict
from src.algorithms import obj_dict as obj_dict

import os
import yaml



#create the obj
class poly_obj_composite():
    def __init__(self,poly_to_add,degree_dict, weights_dict, bias_dict, x_mins_dict, ratios_dict, noise):
        self.noise_ = noise
        self.weights_dict_ = weights_dict
        self.bias_dict_ = bias_dict
        self.ratios_dict_ = ratios_dict
        self.poly_to_add_ = poly_to_add #names of the polynomials to add
        self.x_mins_dict_ = x_mins_dict #just to know the degree of each polynomials
        self.ndims_ = np.array([len(x_mins_dict[key]) for key in poly_to_add]).sum()
        self.degree_dict_ = degree_dict
    def evaluate(self, X):
        z_dict = {}
        first_col= 0
        for fname in self.poly_to_add_:
            poly_ndims = len(self.x_mins_dict_[fname])
            degree = self.degree_dict_[fname]
            mk_poly_features = PolynomialFeatures(degree)
            last_col = first_col+poly_ndims
            poly_features = mk_poly_features.fit_transform(X[:,first_col:last_col])
            z_dict[fname] = self.ratios_dict_[fname]*(poly_features@self.weights_dict_[fname] + self.bias_dict_[fname])
            first_col = last_col
        z = np.array(list(z_dict.values())).sum(axis=0)
        return z
    def sample(self, X):
        return self.evaluate(X) + np.random.normal(0,self.noise_, X.shape[0])[:,np.newaxis]
    def evaluate_1D(self, x, dim=0, other_dim_val=0):
        # Est-ce que je peux isoler les poids d'une dimension???
        X = np.hstack([x if i==dim else np.full(x.shape, other_dim_val) for i in range(self.ndims_)])
        
        return self.evaluate(X)
    def sample_1D(self, x, dim=0, other_dim_val=0):
        X = np.hstack([x if i==dim else np.full(x.shape, other_dim_val) for i in range(self.ndims_)])
        return self.sample(X)

with open("poly4d_FWHM_BLEACH.pkl", 'rb') as f:
    poly4d_FWHM_BLEACH = pickle.load(f)
obj_dict["poly4d_FWHM_BLEACH.pkl"] = poly4d_FWHM_BLEACH


def get_X_domain(ndims, n_points_default = 20, x_min_default=-1, x_max_default=1, x_mins=None, x_maxs=None, n_points=None):
    if x_mins is None:
        x_mins   = [x_min_default]*ndims
    if x_maxs is None:
        x_maxs   = [x_max_default]*ndims
    if n_points is None:
        n_points = [n_points_default]*ndims
    grids = np.meshgrid(*[np.linspace(x_mins[i], x_maxs[i], n_points[i]) for i in range(ndims)])
    X = np.hstack([grid.ravel()[:,np.newaxis] for grid in grids])
    return X

def discrete_random_GS(X, n_samples=100):
    return np.vstack([X[np.random.randint(0, X.shape[0])] for i in range(n_samples)])

def continuous_random_GS(x_min=-1, x_max=1, ndims=4, n_samples=100):
    x_mins = [x_min]*ndims
    x_maxs = [x_max]*ndims
    return np.array([np.random.uniform(low=x_mins, high=x_maxs) for i in range(n_samples)])
    

def normal_pdf(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sigma)**2)


#################################################
## Analysis functions
#################################################
def dict_from_optim_data(save_folders, nb_samples=None):


    optim_data = {folder:{} for folder in save_folders}
    for folder in save_folders:
        with open(os.path.join(folder, "config.yml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        optim_data[folder]["config"] = config
        
        if 'regressor_name' in config:
            algo = TS_sampler(regressor_name_dict[config['regressor_name']](**config['regressor_args']))
        #The other cases are her to support earlier configuration files (before march 30 2020)
        elif config['kernel_str'] is not None:
            algo = algorithms.custom_GP_TS(config["kernel_str"], config["alpha"])
        elif config['poly_deg'] is not None:
            algo = algorithms.poly_reg_TS(algorithms.linear_regression(config["poly_noise"], config["poly_reg"]), config["poly_deg"])

        optim_data[folder]['algo'] = algo
        X_list, y_list, dts_sampling_list, dts_update_list = [], [], [], []
    #     for i in range(config["nbre_trials"]):
        if not nb_samples:
            nb_samples = config["nbre_trials"]
        for i in range(nb_samples):
            if os.path.exists(os.path.join(folder, f"X_{i}.csv")):
                X_list.append(np.loadtxt(os.path.join(folder, f"X_{i}.csv"), delimiter=','))
                y_list.append(np.loadtxt(os.path.join(folder, f"y_{i}.csv"), delimiter=','))
                dts_sampling_list.append(np.loadtxt(os.path.join(folder, f"dts_sampling_{i}.csv")))
                dts_update_list.append(np.loadtxt(os.path.join(folder, f"dts_update_{i}.csv")))
            else:
                print(f"The {i}th iteration is not preset for {folder}")
                break
#        print(np.array(X_list).mean())
        optim_data[folder]['X_list'] =  X_list
        optim_data[folder]['y_list'] = y_list
        optim_data[folder]['dts_sampling_list'] = dts_sampling_list
        optim_data[folder]['dts_update_list'] = dts_update_list
    return optim_data


#def get_regret_data(optim_data, obj, nb_samples=None):
#    save_folders = list(optim_data.keys())
#    config = optim_data[save_folders[0]]['config']
#    if not nb_samples:
#        nb_samples = config["nbre_trials"]
#
#    regrets_list = []
#    cumul_regrets_list = []
#
#    ndims = optim_data[save_folders[0]]['config']['ndims']
#
#    regret_data = {}
#    level1_keys = ["GS"] + save_folders
#    level2_keys = ("regret", "cumul_regret")
#    regret_data = {}
#    for key1 in level1_keys:
#        regret_data[key1] = {}
#        for key2 in level2_keys:
#            regret_data[key1][key2] = []
#
#
#    X = get_X_domain(ndims=ndims, n_points_default=40)
#    if ndims==1:
#        obj_max = np.max(obj.evaluate_1D(X))
#    else:
#        obj_max = np.max(obj.evaluate(X))
#    # for i in range(config["nbre_trials"]):
#    for i in range(nb_samples):
#    #     regrets = obj_max - obj.evaluate_1D(X_list[i][:,np.newaxis])
#    #     regret_data["GP"]["regret"].append(regrets)
#    #     regret_data["GP"]["cumul_regret"].append(np.cumsum(regrets))
#        if ndims==1:
#            regrets = obj_max - obj.evaluate_1D(continuous_random_GS(-1,1,n_samples=config["optim_length"])[:,np.newaxis])
#        else:
#            regrets = obj_max - obj.evaluate(continuous_random_GS([-1]*ndims,[1]*ndims,n_samples=config["optim_length"]))
#        regret_data["GS"]["regret"].append(regrets)
#        regret_data["GS"]["cumul_regret"].append(np.cumsum(regrets))
#
#
#        for folder in save_folders:
#            if os.path.exists(os.path.join(folder, f"X_{i}.csv")):
#                X = get_X_domain(ndims=ndims, n_points_default=optim_data[folder]['config']['n_points_default'])
#                if ndims==1:
#                    obj_max = np.max(obj.evaluate_1D(X))
#                    regrets = obj_max - obj.evaluate_1D(optim_data[folder]['X_list'][i][:,np.newaxis])
#                else:
#                    obj_max = np.max(obj.evaluate(X))
#                    regrets = obj_max - obj.evaluate(optim_data[folder]['X_list'][i])
#                regret_data[folder]["regret"].append(regrets)
#                regret_data[folder]["cumul_regret"].append(np.cumsum(regrets))
#    return regret_data
    
def get_regret_data(optim_data):
    
    regret_data = {}
    save_folders = list(optim_data.keys())
    
    
#    if not nb_samples:
#        nb_samples = config["nbre_trials"]
    
    
    
    for folder in save_folders:
        regret_data[folder] = {}
        config = optim_data[folder]['config']
        nb_samples = config['nbre_trials']
        obj = obj_dict[config['obj_name']]
        ndims = config['ndims']
        n_points_default = config['n_points_default']
        X_grid = get_X_domain(ndims=ndims, n_points_default=n_points_default)
        if ndims==1:
            obj_max = np.max(obj.evaluate_1D(X_grid))
            regrets = [obj_max - obj.evaluate_1D(optim_data[folder]['X_list'][i][:,np.newaxis]) for i in range(nb_samples)]
        else:
            obj_max = np.max(obj.evaluate(X_grid))
            regrets = [obj_max - obj.evaluate(optim_data[folder]['X_list'][i]) for i in range(nb_samples)]
        cumul_regrets = [np.cumsum(regrets[i]) for i in range(nb_samples)]
        regret_data[folder]["regret"] = regrets
        regret_data[folder]["cumul_regret"] = cumul_regrets
        if ndims==1:
            obj_max = np.max(obj.evaluate_1D(X_grid))
            regrets = [obj_max - obj.evaluate_1D(discrete_random_GS(X_grid, n_samples=config["optim_length"])[:,np.newaxis]) for i in range(nb_samples)]
        else:
            obj_max = np.max(obj.evaluate(X_grid))
            regrets = [obj_max - obj.evaluate(discrete_random_GS(X_grid, n_samples=config["optim_length"])) for i in range(nb_samples)]
        cumul_regrets = [np.cumsum(regrets[i]) for i in range(nb_samples)]
        regret_data[folder]["regret_GS"] = regrets
        regret_data[folder]["cumul_regret_GS"] = cumul_regrets

        
        
    
    return regret_data

#def plot_regret(regret_data, optim_data,save_folders, surnames_dict, colors_dict=None, line_styles_dict=None, show_std=True, plot_all_curves=False):
#    config = optim_data[save_folders[0]]['config']
#
#    arr = np.array([len(optim_data[key]["X_list"]) for key in optim_data])
#    all_same_repeats = np.all(arr == arr[0])
#
#
#    for algo in regret_data.keys():
#
#        matrix = np.squeeze(np.array(regret_data[algo]["cumul_regret"]))
#        mean, std = matrix.mean(axis=0), matrix.std(axis=0)
##        print(algo,matrix.shape, mean.shape, std.shape)
#
#
#        if algo in surnames_dict:
#            label=surnames_dict[algo]
#        else:
#            label = algo
#        if not all_same_repeats:
#            label = label+f" (n={matrix.shape[0]})"
#        if algo != 'GS':
#            config = optim_data[algo]['config']
#        x = range(1, len(mean)+1)
#
#        if line_styles_dict is not None:
#            line_style = line_styles_dict[algo]
#        else:
#            line_style = '-'
#        if colors_dict is not None:
#            line = plt.plot(x, mean,line_style,color=colors_dict[algo], label=label)
#        else:
#            line = plt.plot(x, mean, line_style, label=label)
#        if show_std:
#            plt.fill_between(x, mean-std, mean+std, alpha=0.1, color=line[0].get_color())
#        if plot_all_curves:
#            if algo !='GS':
#                plt.plot(matrix.T, alpha=0.2, color=line[0].get_color())
#    plt.ylabel("Cumulative pseudo-regret")
#    plt.xlabel("Sample number")
#    plt.legend(framealpha=0.6)
    
    
def plot_regret(regret_data, optim_data,save_folders, surnames_dict, colors_dict=None, line_styles_dict=None, show_std=True, specific_folder=None, plot_all_curves=False):
    
    
    arr = np.array([len(optim_data[key]["X_list"]) for key in optim_data])
    all_same_repeats = np.all(arr == arr[0])
    
    
    for algo in regret_data.keys():
        if (specific_folder is None) or (specific_folder ==algo):
            config = optim_data[algo]['config']
            for i in (0,1):
                if i==1:
                    matrix = np.squeeze(np.array(regret_data[algo]["cumul_regret"]))
                    if algo in surnames_dict:
                        label=surnames_dict[algo]
                    else:
#                        label = algo
#                    if not all_same_repeats:
#                        label = label+f" (n={matrix.shape[0]})"
                        label="TS"
                else:
                    matrix = np.squeeze(np.array(regret_data[algo]["cumul_regret_GS"]))
                    label='GS'
                mean, std = matrix.mean(axis=0), matrix.std(axis=0)
        #        print(algo,matrix.shape, mean.shape, std.shape)
            
                    
    #            if algo != 'GS':
    #                config = optim_data[algo]['config']
                x = range(1, len(mean)+1)
                
                if line_styles_dict is not None:
                    line_style = line_styles_dict[algo]
                else:
                    line_style = '-'
                if colors_dict is not None:
                    line = plt.plot(x, mean,line_style,color=colors_dict[algo], label=label)
                else:
                    line = plt.plot(x, mean, line_style, label=label)
                if show_std or i==0:
                    plt.fill_between(x, mean-std, mean+std, alpha=0.1, color=line[0].get_color())
                if plot_all_curves:
                    if i == 1:
                        plt.plot(matrix.T, alpha=0.2, color=line[0].get_color())
    plt.ylabel("Cumulative pseudo-regret")
    plt.xlabel("Sample number")
    plt.legend(framealpha=0.6)
    
    


def show_slices_nd(objective, optim_data=None, folder=None, X_sample=None, y_sample=None, algo=None, surnames_dict={}, idx=0, nb_iters=None, ax = None):
    if ax:
        plt.sca(ax)
    
#    save_folders = list(optim_data.keys())
    
    if optim_data is not None:
        X_sample = optim_data[folder]['X_list'][0]
        y_sample = optim_data[folder]['y_list'][0]
        algo = optim_data[folder]['algo']
    
    if nb_iters:
        X_sample = X_sample[:nb_iters]
        y_sample = y_sample[:nb_iters]
    
    
    algo.update(X_sample, y_sample)
    

    discrete_divs=7

    import inflect
    p = inflect.engine()

    # n_samples = 100
    # noise_std = 0.1
    X = get_X_domain(X_sample.shape[1], n_points_default = 40)
    z = objective.evaluate(X)
    x_max = X[np.argmax(z)]
#     ndims = obj.ndims_
    ndims=X_sample.shape[1]
    x_min_default = -1
    x_max_default = 1
    x_mins = [x_min_default]*ndims
    x_maxs = [x_max_default]*ndims
    ylims = (-0.1, 1.1)

    n_points = 4
    ncols = X_sample.shape[1]
    nrows = n_points #np.ceil(n_plots/ncols).astype(int)
    figsize = np.array(matplotlib.rcParams['figure.figsize'])*np.array([ncols,nrows])/1.7
    #figsize = figsize*8/figsize[0]
    fig = plt.figure(figsize=figsize)
    i_plot = 1
    for i_point in range(n_points):

        # index of a point at regular distance from the optima
        optim_dists = np.sqrt(((X_sample-x_max)**2).sum(axis=1))
        dists_range = np.linspace(optim_dists.min(), optim_dists.max(), n_points)
        index = (np.abs(optim_dists-dists_range[i_point])).argmin()
        point_x = X_sample[index]


        for dim in range(X_sample.shape[1]):
            x_space = [np.array([point_x[i]]) if i!=dim else np.linspace(x_mins[i], x_maxs[i]) for i in range(ndims)]
            grids = np.meshgrid(*x_space)
            X = np.hstack([grid.ravel()[:,np.newaxis] for grid in grids])
    #         X = get_X_domain(obj.ndims_, n_points_default = 10)

        #     poly_features = np.concatenate([mk_poly_features.fit_transform(X[:,:2]), mk_poly_features.fit_transform(X[:,2:4])],axis=1)
        #     z = poly_features@coeffs[:,np.newaxis]
            z = objective.evaluate(X)

            mean, std = algo.predict(X )
            # std = np.diagonal(std)
            plt.subplot(nrows,ncols,i_plot); i_plot+=1
            line = plt.plot(X[:,dim], mean, label="mean")
            plt.plot(X[:,dim], z, label="obj", color="black")
            plt.fill_between(X[:,dim], mean.flatten()-std.flatten(), mean.flatten()+std.flatten(), alpha=0.1, color=line[0].get_color(), label='std')
            plt.scatter(point_x[dim], y_sample[index], c='black', s=10, label='data point')
            certer_dist = np.sqrt((np.array(point_x)[np.arange(len(point_x)) != dim]**2).sum())
            optim_dist = np.sqrt(((np.array(point_x)[np.arange(len(point_x)) != dim] - x_max.flatten()[np.arange(len(point_x)) != dim])**2).sum())
            plt.vlines(x_max[dim], ylims[0], ylims[1], alpha=0.6)
            plt.ylim(ylims)
            plt.yticks([])
            if dim==0:
                plt.ylabel(f"d = {optim_dists[index]:.1f} ")
                plt.yticks([0,1])
            plt.yticks([0,1])

            if i_point<n_points-1:
                plt.xticks(np.linspace(-1,1,discrete_divs))
            else:
                plt.xticks(np.linspace(-1,1,discrete_divs))
                plt.xlabel(p.ordinal(dim+1)+" param")
            xlim = plt.xlim()
            if i_plot in (2,):
                plt.legend()
            plt.grid()
    plt.tight_layout()
    
def plot_optim_1D(obj, algo,X_list, y_list, config=None, idx=0, title=None, ax=None, max_iter=None, time_color=True, update=True, func_samples=0):
#     plt.figure(figsize=np.array(matplotlib.rcParams["figure.figsize"])*1.5)
#     algo = algorithms.custom_GP_TS(config["kernel_str"], config["alpha"])
    X_results = X_list[idx]
    if len(X_results.shape)==1:
        X_results = X_results[:,np.newaxis]
    if max_iter is None:
        max_iter = len(X_list[idx])
    
    if ax:
        plt.sca(ax)
    if update:
        algo.update(X_results[:max_iter], y_list[idx][:max_iter])
    X = get_X_domain(ndims=1, n_points_default=40)
    y = obj.evaluate_1D(X)
#    try:
#    import pdb; pdb.set_trace()
#    print(algo)
#    print(algo.regressor)
    mean, std = algo.predict(X)
#    except ValueError:
#        mean, std = algo.predict(X, return_std=True)


    plt.plot(X, y, color='black', label='ground truth')
    
    if time_color:
        plt.scatter(X_results[:max_iter], y_list[idx][:max_iter], c=np.arange(len(y_list[idx][:max_iter])), s=15, alpha=1, label='Observations')
    else:
        plt.scatter(X_results[:max_iter], y_list[idx][:max_iter], c='black', s=15, alpha=0.5, label='Observations')
    
    line = plt.plot(X,mean, label='mean $\pm$ std')
    
    plt.fill_between(X.flatten(), mean.flatten()-std, mean.flatten()+std, alpha=0.1, color=line[0].get_color() )
    y_range = y.max() - y.min()
    plt.ylim(y.min()-y_range*0.3, y.max()+y_range*1)
    if config is not None:
        nb_xticks = config["n_points_default"]
        if nb_xticks<=10:
            plt.xticks(np.linspace(config['x_min_default'], config['x_max_default'], nb_xticks))
            
    if func_samples:
        for i in range(func_samples):
            plt.plot(X, algo.sample(X) ,alpha=0.1)
    ax = plt.gca()
    # ax.xaxis.grid()
    
#    if title:
#        plt.title(title)
#    else:
#        plt.title(f"{len(X_sample)} observations")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(framealpha=0.6)
    if time_color:
        plt.colorbar()
        

#     if 'gp' in dir(algo):
#         print(gp)

#     print(algo)
#     plt.show()
