from . import split3, microscope
split_sted = split3.split_sted
import skimage.io as skio
import numpy as np
import functools
import os
import json
import warnings
# import matplotlib; matplotlib.use("TkAgg")



def acquire(p_ex, p_sted, nb_images, tau_exc, config_conf, config_sted):
    ##### PARAMETERS TO CHANGE BY THE USER  ################
    #save directory
#    directory = "2020-09-09_split3valid/"
#    fname = "PSD95_sted10_exc20"


    #sted stack params
    config_sted_stack = {"Tau/Exc": tau_exc, #fixed exctation power if np.inf
                        "Nb/Images": nb_images, #number of images in the stack
                        "STED/Power": p_sted, #maximal sted power
                        "Exc/Power": p_ex,} #excitation power <- I recommend to tune this paramter - Albert.
    #######################################################



#   split sted params
    config_split ={
    #Parameter in option in the matlab code
#    "Tg" : 6, #% 'First frame to sum:'
    "Nb_to_sum" : 1, #  The Tg infered from this variable override Tg
    "smooth_factor" : 0.2, #% 'Smoothing factor:'
    "im_smooth_cycles" : 0, #% 'Smoothing cycles image:'
    "phasor_smooth_cycles" : 1, #% 'Smoothing cycles phasor:'
    "foreground_threshold" : 15, #% 'Threshold value:'
    "tau_exc" : config_sted_stack["Tau/Exc"], #% 'Tau_exc'
    "intercept_at_origin" : False, #% 'Fix Intercept at origin'

    #Parameters that are calculated in th matlab code but that could be modified manually
    "M0" : None,
    "Min" : None,

    #Paramaters that are fixed in the matlab code
    "m0" : 1,
    "harm1" : 1, #MATLAB code: harm1=1+2*(h1-1), where h1=1
    "klog" : 4,
    }


    #---------------------- Acquire sted_stack ------------------------
    # config_conf = microscope.get_config("Setting confocal configuration.", "conf_logo.png")
    # config_sted = microscope.get_config("Setting STED configuration.", "sted_logo.png")
    # verify that confocal and STED configurations can be used together
    # assert microscope.get_imagesize(config_conf) == microscope.get_imagesize(config_sted),\
    #     "Confocal and STED images must have the same size!"


    params_set = {"Dwelltime": microscope.set_dwelltime,
                  "Exc/Power": functools.partial(microscope.set_power, laser_id=3),
                  "STED/Power": functools.partial(microscope.set_power, laser_id=6),
                  "Line_Step": functools.partial(microscope.set_linestep, step_id=0),
                  "Rescue/Signal_Level": functools.partial(microscope.set_rescue_signal_level, channel_id=0),
                  "Rescue/Strength": functools.partial(microscope.set_rescue_strength, channel_id=0)}




    sted_stack=[]
    # conf_stack=[]
    tau_exc = config_sted_stack["Tau/Exc"]
    Nb_images=config_sted_stack["Nb/Images"]
    # conf, _= microscope.acquire(config_conf
    # conf_stack.append(conf[-1][0])
    params_set["Exc/Power"](config_sted, config_sted_stack["Exc/Power"])
    for jm1 in range(Nb_images):

        params_set["STED/Power"](config_sted, jm1*config_sted_stack["STED/Power"]/(Nb_images-1))
        if tau_exc >0:
            print(jm1)
            #params_set["Exc/Power"](config_sted, config_sted_stack["Exc/Power"]*np.exp(-jm1/tau_exc))
        else:
            params_set["Exc/Power"](config_sted, config_sted_stack["Exc/Power"]*np.exp(-jm1/tau_exc)/np.exp(-(Nb_images-1)/tau_exc))
        sted, totaltime = microscope.acquire(config_sted)
        sted_stack.append(sted[-1][0])
    conf, _= microscope.acquire(config_conf)
    # conf_stack.append(conf[-1][0])


    sted_stack = np.array(sted_stack)

    #----------------------- SPLIT STED ---------------------------


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Fin  = split_sted(np.moveaxis(np.array(sted_stack), 0, -1), config_split,  **config_split )
    #Fin = split_sted(sted_stack, config_split,  **config_split, return_analysis_fig=False )



    return Fin, sted_stack
