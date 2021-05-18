import banditopt.optim_microscope as optim
import numpy as np

#x_mins=[(4e-3)*2**-2, (1e-5)*2**-2]
#x_maxs=[(4e-3)*2**2, (1e-5)*2**2]

config =  dict(
    save_folder="split-sted_debug",
    regressor_name="sklearn_BayesRidge",
    regressor_args= {
        "default":{
            "alpha_1": 1e-6,
            "alpha_2": 1e-6,
            "alpha_init": 100.11183053174646,
            "compute_score": True,
            "degree": 4,
            "fit_intercept": True,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
            "lambda_init": 1.4595412992911199,
            "tol": 1.0e-06,
        },
        "SNR":{"alpha_init":1./0.5**2, "lambda_init":1/(2*15/(0.028)**2)**2,},
        "Bleach":{"alpha_init":1./0.06**2, "lambda_init":1/(2/(0.028)**2)**2,},
        "Resolution":{"alpha_init":1./3.**2, "lambda_init":1/(2*100/(0.028)**2)**2,},
    },
    n_divs_default = 25,
    param_names = ["p_sted"],
    with_time=True,
    default_values_dict=dict(
        molecules_disposition = None,
        im_size_nm  = 1000,
        nb_molecules_per_point_src = 50,
        nb_pt_src = 8,
        pixelsize = 20e-9,
        p_ex = 50e-6,
        p_sted = 3e-6,
        dwelltime = 10e-6,
        bleach = True,
        noise = True,
        background=5/10e-6,
        darkcount=0
    ),
    params_conf = { 'p_ex':20e-6, 'p_sted':0, },
    x_mins=[0],
    x_maxs=[90],
    obj_names=["Resolution", "Bleach", ],
    optim_length = 50,
    nbre_trials = 3,
#    borders = [(0,300), (0,1), (0,50)],
    borders = None,
    pareto_only = False,
    split_sted_params = {"p_ex":5, 'p_sted':6.5, 'nb_images':5, 'tau_exc':np.inf}
              )

optim.run_TS(config=config, **config)
