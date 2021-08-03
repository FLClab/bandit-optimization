import banditopt.optim_microscope as optim

x_mins=[5, 5, 10e-6]
x_maxs=[95, 95, 500e-6]

degree = 3

borders = {"Resolution":(0,350), "Bleach":(0,1), "SNR":(0,40)}


config =  dict(
    save_folder=f"2020-07-12-4params/deg{degree}",
    regressor_name="sklearn_BayesRidge",
    regressor_args= {
        "default":{
            "degree": degree,
            'param_space_bounds':[(x_mins[i], x_maxs[i]) for i in range(len(x_mins))],
        },
        "Resolution":{"alpha_init":1./3.**2, "lambda_init":1/(2*100/(0.028)**2)**2,},
        "Bleach":{"alpha_init":1./0.06**2, "lambda_init":1/(2/(0.028)**2)**2,},
        "SNR":{"N0_w":1, "std0_w":borders[0], N0_n:1, N0_w:},
    },
     n_divs_default = 20, #20**3 = 8000
#    n_points = [10, 10, 10, 10],
    param_names = ["p_ex", "p_sted", "dwelltime", "line_step"],
    # param_names = ["p_ex"],
    with_time=False,
    default_values_dict=dict(
        # molecules_disposition = None,
        # im_size_nm  = 1000,
        # nb_molecules_per_point_src = 50,
        # nb_pt_src = 8,
        pixelsize = 20e-9, # I verified that the real pixel size is 20e-9, also, 10 radom points initially hardcoded
        # p_ex = 50e-6,
        # p_sted = 3e-6,
        # dwelltime = 10e-6,
        # bleach = True,
        # noise = True,
        # background=5/10e-6,
        # darkcount=0
    ),
    params_conf = None,
    x_mins=x_mins,
    x_maxs=x_maxs,
    obj_names=["Resolution", "Bleach", 'SNR'],
    # obj_names=["SNR", "Bleach", ],
    optim_length = 150,
    nbre_trials = 3,
    borders = borders,
    # borders = None,
    pareto_only = True,
    time_limit = 500e-6 #total_time/nbre_pixels
    NSGAII_kwargs = dict(NGEN=250, MU=200, L = 40, min_std=np.sqrt(2e-4*len(obj_names)))
              )

optim.run_TS(config=config, **config)

