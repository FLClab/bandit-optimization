import numpy as np
import matplotlib.pyplot as plt
from banditopt.optim import regressors_dict
import plotly.graph_objects as go
import pandas as pd
import random as python_random

# from .hmc import monte_carlo_pred


def slice_plot_data(config, X, Y, point_idx=0, axis=0, obj=0, n_samples=100, ndivs=100):
    regressor = regressors_dict[config["regressor_name"]](**config["regressor_args"]["default"])
    regressor.update(X,Y[:, obj][:,np.newaxis])
    xmin = config['x_mins'][axis]
    xmax = config['x_maxs'][axis]
    x_default = X[point_idx,:].flatten()
    X_test = np.stack([np.linspace(xmin, xmax, ndivs) if i==axis
                  else np.array(list([val]*ndivs))
                  for i, val in enumerate(x_default)], axis=1)
    x = X_test[:,axis]
    mean, std, std_noisy = regressor.get_mean_std(X_test, return_withnoise=True)
    samples_list = []
    for i in range(n_samples):
        samples_list.append(regressor.sample(X_test))
    x_data = [X[point_idx,axis]]
    y_data = [Y[point_idx,obj]]
    return x, mean, std, std_noisy, samples_list, x_data, y_data


def HCMC_slice_plot_data(model, weight_samples_dict, rng_key, rng_key_predict, config, X, Y, point_idx=0, axis=0, obj=0, n_samples=100, ndivs=100):

    xmin = config['x_mins'][axis]
    xmax = config['x_maxs'][axis]
    x_default = X[point_idx,:].flatten()
    X_test = np.stack([np.linspace(xmin, xmax, ndivs) if i==axis
                  else np.array(list([val]*ndivs))
                  for i, val in enumerate(x_default)], axis=1)
    x = X_test[:,axis]

    mean, percentiles, percentiles_noisy, samples = monte_carlo_pred(model, weight_samples_dict, rng_key, rng_key_predict, X_test)

    samples_list = python_random.sample(samples, n_samples)
    x_data = [X[point_idx,axis]]
    y_data = [Y[point_idx,obj]]
    return x, mean, percentiles, percentiles_noisy, samples_list, x_data, y_data


#def HCMC_slice_plot_data(func, weight_samples_dict, config, X, Y, point_idx=0, axis=0, obj=0, n_samples=100, ndivs=100):
#    # TODO: noise samples should be used instead
#
#    xmin = config['x_mins'][axis]
#    xmax = config['x_maxs'][axis]
#    x_default = X[point_idx,:].flatten()
#    X_test = np.stack([np.linspace(xmin, xmax, ndivs) if i==axis
#                  else np.array(list([val]*ndivs))
#                  for i, val in enumerate(x_default)], axis=1)
#    x = X_test[:,axis]
#    samples_list = []
#    samples_df = pd.DataFrame(weight_samples_dict).iloc[:,1:]
#    for j in range(len(samples_df)):
#        samples_list.append(
#        func(*[X_test[:,i] for i in range(X_test.shape[1])], **samples_df.loc[j].to_dict())
#        )
#    mean = np.concatenate([sample[:,np.newaxis] for sample in samples_list], axis=1).mean(axis=1)
#    std = np.concatenate([sample[:,np.newaxis] for sample in samples_list], axis=1).std(axis=1)
##    percentiles = np.percentile(np.concatenate([sample[:,np.newaxis] for sample in samples_list], axis=1), [2.25, 97.75], axis=1)
##    std = percentiles # THis is not the std, those are the low and up percentiles...
#
#    import pdb; pdb.set_trace()
#
#    std_noisy = np.sqrt(std**2 + 1/np.mean(weight_samples_dict['prec_obs']))
#
#    samples_list = python_random.sample(samples_list, n_samples)
#    x_data = [X[point_idx,axis]]
#    y_data = [Y[point_idx,obj]]
#    return x, mean, std, std_noisy, samples_list, x_data, y_data, percentiles

def plot_1dregression(x, mean, std, std_noisy, samples_list, x_data, y_data, legend=True, **kwargs):
    """
    PARAMETERS:
    x: array of values of the domain
    mean: mean estimate of the regression model
    std: standard deviation of the estimate by the regression model
    std_noisy: standard deviation of the estimate (accounting for the noise)
               by the regression model
    samples_list: list of y arrays of some number of sampled functions (100 recommended)
    x_data: x array of observed points
    y_data: y array of observed points

    NOTES: all arrays in input are assumed to be 1d flattened arrays
    """
    plt.plot(x, mean,color='blue', label='Mean')
    if std.shape[0] == 2:
        plt.fill_between(x, std[0,:], std[1,:],  color='blue', alpha=0.25, label='95.5 CI')
        plt.fill_between(x, std_noisy[0,:], std_noisy[1,:],  color='blue', alpha=0.15, label='95.5 CI (noise)')
    else:
        plt.fill_between(x, mean-2*std,mean+2*std,  color='blue', alpha=0.25, label='2*std')
        plt.fill_between(x, mean-2*std_noisy,mean+2*std_noisy,  color='blue', alpha=0.15, label='2*std(noise)')
    argmaxs = []
    for y_sample in samples_list:
        if len(argmaxs) == 0:
            label = f'{len(samples_list)} samples'
        else:
            label = None
        plt.plot(x, y_sample, color='orange', alpha=0.9, linewidth=0.5,zorder=0, label=label)
        argmaxs.append(x[np.argmax(y_sample)],)
    plt.scatter(x_data, y_data, edgecolor="black",color='white',alpha=0.9,zorder=10, label='Datapoint(s)')
    if 'acqu_func' in kwargs:
        ylim = plt.ylim()
        plt.ylim(ylim[0] - (ylim[1]-ylim[0])*0.2, ylim[1])
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        plt.sca(ax2)
        plt.hist(argmaxs, edgecolor='black',label='~P(sampling)', bins=np.linspace(np.min(x), np.max(x), 20))
        ylim = plt.ylim()
        plt.ylim(ylim[0], ylim[0] + (ylim[1]-ylim[0])*5)
        plt.yticks([])
        if legend:
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            first_legend = plt.legend(handles1, labels1, loc = kwargs["acqu_func"]["legend_locs"][0])
            ax2.add_artist(first_legend)
            ax2.legend(handles2, labels2, loc=kwargs["acqu_func"]["legend_locs"][1])
        plt.sca(ax1)
    else:
        if legend:
            plt.legend()
    return plt.gcf()

def plot_ndregression(config, X, Y,  obj=0, X_test=None, Y_test=None, y_bounds=None, hmc_kwargs=None):
    # add x1...xd values
    
    # objs_df = pd.DataFrame(Y, columns=config["obj_names"])
    # data_df = pd.concat([params_df, objs_df], axis=1)
    # params_df["Error"] = objs_df[obj]
    

    if hmc_kwargs is not None:
        data_df = pd.DataFrame(X, columns=config["param_names"])
        data_df['y_obs'] = Y[:,obj]
        
        model, weight_samples_dict, rng_key, rng_key_predict = \
        hmc_kwargs['model'], hmc_kwargs['weight_samples_dict'], hmc_kwargs['rng_key'], hmc_kwargs['rng_key_predict']
        mean, percentiles, percentiles_noisy, samples = monte_carlo_pred(model, weight_samples_dict, rng_key, rng_key_predict, X)

#        mean, std, std_noisy = regressor.get_mean_std(X, return_withnoise=True)
        data_df['y_pred'] = mean
        data_df['Error'] = data_df['y_obs'] - data_df['y_pred']
        plus_minus_transform = -1*(data_df['Error'] <0) + (data_df['Error'] >0)
        data_df['±95.5 CI (noise)'] = percentiles_noisy[1,:] - data_df['y_pred']
        data_df['±95.5 CI (noise)'][data_df['Error'] <0] = (percentiles_noisy[0,:] - data_df['y_pred'])[np.array([data_df['Error'] <0]).flatten()]
        data_df['±95.5 CI'] = percentiles[1,:] - data_df['y_pred']
        data_df['±95.5 CI'][data_df['Error'] <0] = (percentiles[0,:] - data_df['y_pred'])[np.array([data_df['Error'] <0]).flatten()]

    else:

        regressor = regressors_dict[config["regressor_name"]](**config["regressor_args"]["default"])
        regressor.update(X,Y[:, obj][:,np.newaxis])
        if (X_test is not None) and (Y_test is not None):
            X, Y = X_test, Y_test
            
        data_df = pd.DataFrame(X, columns=config["param_names"])
        data_df['y_obs'] = Y[:,obj]

        mean, std, std_noisy = regressor.get_mean_std(X, return_withnoise=True)
        data_df['y_pred'] = mean
        data_df['Error'] = data_df['y_obs'] - data_df['y_pred']
        plus_minus_transform = -1*(data_df['Error'] <0) + (data_df['Error'] >0)
        data_df['±y_std*2 (noise)'] = std_noisy*2 * plus_minus_transform
        data_df['±y_std*2'] = std*2 * plus_minus_transform

    data_df.insert(0, 'im #', data_df.index+1)

    cols = data_df.columns
    if y_bounds is None:
        y_min = data_df[["y_obs","y_pred"]].min().min()
        y_max = data_df[["y_obs","y_pred"]].max().max()
    else:
        y_min = y_bounds[0]
        y_max = y_bounds[1]
    
    xmins = [0] + config['x_mins'] + \
            [y_min]*2 + \
            [-(y_max-y_min)/2]*3
    #         [-np.abs(data_df[["Error","±y_std*2 (noise)"]]).max().max()]*3
    xmaxs = [len(data_df)+1] + config['x_maxs'] + \
            [y_max]*2 + \
            [+(y_max-y_min)/2]*3
    #         [np.abs(data_df[["Error","±y_std*2 (noise)"]]).max().max()]*3

    dimensions = [dict(range=[xmins[i],xmaxs[i]], label=col, values=data_df[col]) for i, col in enumerate(cols)]

    color = 'y_obs'
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = data_df[color],
                       colorscale = 'Jet',
                       showscale = True,
                       cmin = data_df[["y_obs","y_pred"]].min().min(),
                       cmax = data_df[["y_obs","y_pred"]].max().max(),
                       colorbar=dict(title=color)),
            dimensions = dimensions, ),
    )


    fig.update_layout(title=f'{config["obj_names"][obj]}')
    return fig, data_df
