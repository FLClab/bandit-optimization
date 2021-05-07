import src.optim as optim

config =  dict(
    save_folder="debug_trialX",
    regressor_name="sklearn_BayesRidge",
    regressor_args= {
        "default":{
            "alpha_1": 5000.0,
            "alpha_2": 49.94414719461602,
            "alpha_init": 100.11183053174646,
            "compute_score": True,
            "degree": 4,
            "fit_intercept": True,
            "lambda_1": 5000.0,
            "lambda_2": 3425.7338263935626,
            "lambda_init": 1.4595412992911199,
            "tol": 1.0e-06,
        },
        "SNR":{"alpha_init":1./0.5**2, "lambda_init":1/(2*15/(0.004 - 0.0005)**2)**2, "alpha_2": 5000/(1./0.5**2), "lambda_2": 5000./(1/(2*15/(0.004 - 0.0005)**2)**2),},
        "Bleach":{"alpha_init":1./0.06**2, "lambda_init":1/(2/(0.004 - 0.0005)**2)**2, "alpha_2": 5000/(1./0.03**2), "lambda_2": 5000./(1/(2/(0.004 - 0.0005)**2)**2),},
        "Resolution":{"alpha_init":1./3.**2, "lambda_init":1/(2*100/(0.004 - 0.0005)**2)**2, "alpha_2": 5000./(1/3**2), "lambda_2": 5000./(1/(2*100/(0.004 - 0.0005)**2)**2),},
    },
    n_divs_default = 25,
    param_names = ["p_sted", "dwelltime"],
    with_time=True,
    default_values_dict=dict(
        molecules_disposition = None,
        im_size_nm  = 1000,
        nb_molecules_per_point_src = 30,
        nb_pt_src = 8,
        pixelsize = 20e-9,
        p_ex = 1e-3,
        p_sted = 3e-6,
        dwelltime = 10e-6,
        bleach = True,
        noise = True,
        background=5/10e-6,
        darkcount=0
    ),
    params_conf = { 'p_ex':100e-6, 'p_sted':0, 'dwelltime':1e-5*100,},
    x_mins=[4e-3*2**-3, 100*1e-5*2**-3],
    x_maxs=[4e-3, 100*1e-5*2**3],
    obj_names=["Resolution", "Bleach", "SNR", ],
    optim_length = 40,
    nbre_trials = 1,
#    borders = [(0,300), (0,1), (0,50)],
    borders = None
              )
    

optim.run_TS(config=config, **config)

