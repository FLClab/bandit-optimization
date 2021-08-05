import banditopt.optim_microscope as optim
import numpy as np

x_mins=[0.2, 1, 2e-6, 0.5]
x_maxs=[40, 75, 100e-6, 10.5]



degree = 3

obj_names=["Resolution", "Bleach", 'SNR']
borders = {"Resolution":(0,350), "Bleach":(0,1), "SNR":(0,6)} #WARNING: this should be in the same order as obj_names


param_names = ["p_ex", "p_sted", "dwelltime", "line_step"]

config =  dict(
    save_folder=f"../bandit-optimization-experiments/2020-08-04-mult_params/params{len(x_mins)}_deg{degree}_{''.join([name[0] for name in param_names])}_{''.join([name[0] for name in obj_names])}",
    regressor_name="sklearn_BayesRidge",
    regressor_args= {
        "default":{
            "degree": degree,
            'param_space_bounds':[(x_mins[i], x_maxs[i]) for i in range(len(x_mins))],
        },
        "Resolution":{"N0_w":1, "std0_w":borders["Resolution"][1]-borders["Resolution"][0], "N0_n":1, "std0_n":(borders["Resolution"][1]-borders["Resolution"][0])/5},
        "Bleach":{"N0_w":1, "std0_w":borders["Bleach"][1]-borders["Bleach"][0], "N0_n":1, "std0_n":(borders["Bleach"][1]-borders["Bleach"][0])/5},
        "SNR":{"N0_w":1, "std0_w":borders["SNR"][1]-borders["SNR"][0], "N0_n":1, "std0_n":(borders["SNR"][1]-borders["SNR"][0])/5},
    },
    param_names = param_names,
    with_time=False,
    default_values_dict=dict(
        p_ex = 4,
        p_sted = 35,
        dwelltime = 20e-6,
        line_step = 3,
        pixelsize = 20e-9,
    ),
    params_conf = None,
    x_mins=x_mins,
    x_maxs=x_maxs,
    obj_names=obj_names,
    # obj_names=["SNR", "Bleach", ],
    optim_length = 150,
    nbre_trials = 5,
    borders = list(borders.values()),
    # borders = None,
    pareto_only = True,
    time_limit = 500e-6, #total_time/nbre_pixels
    # time_limit = None,
#    NSGAII_kwargs = dict(NGEN=250, MU=100, L = 40, min_std=np.sqrt(2e-4*len(obj_names)))
    NSGAII_kwargs = dict(NGEN=250, MU=100, L = 40, min_std=np.sqrt(2e-4*len(obj_names)), integer_params=['line_step'], param_names=param_names)

    # pareto_option = None,
              )

optim.run_TS(config=config, **config)
