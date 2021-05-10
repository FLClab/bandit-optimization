import numpy as np
from pysted import base
import time
#The larger the pixel size, the smaller the "microscope.cache" time.
#The larger the image, the larger the "microscope.get_signal_and_bleach_fast" time.




def simulate_image(molecules_disposition = None, im_size_nm  = 1000, nb_molecules_per_point_src = 100, nb_pt_src = 4,
                   pixelsize = 10e-9, p_ex = 1e-6, p_sted = 30e-3, dwelltime = 10e-6,
                   bleach = True, noise = True, background=0, darkcount=0):

#    dwelltime  = dwelltime/100
    if molecules_disposition is None:
        im_size_pixels = np.rint(im_size_nm*1e-9/pixelsize).astype(int)
        im_shape = (im_size_pixels, im_size_pixels)


        molecules_disposition = np.zeros(im_shape)
        molecules_disposition[np.random.randint(im_size_pixels, size=nb_pt_src), np.random.randint(im_size_pixels, size=nb_pt_src)] = nb_molecules_per_point_src


    print("Setting up the microscope ...")
    # Microscope stuff
    egfp = {"lambda_": 535e-9,
            "qy": 0.6,
            "sigma_abs": {488: 1.15e-20, #initially 488
                          640: 1.15e-20,
                          1000: 1.15e-20,
                          575: 6e-21, #initially 575
                          775: 6e-21,},

            "sigma_ste": {560: 1.2e-20,
                          575: 6.0e-21,
                          775: 6.0e-21,
                          580: 5.0e-21},
            "sigma_tri": 1e-21,
            "tau": 3e-09,
            "tau_vib": 1.0e-12,
            "tau_tri": 5e-6,
            "phy_react": {488: 2.5e-8,   # 1e-4
                          640: 1e-7,
                          1000: 2.5e-8,
                          575: 3e-9, # 1e-8
                          775: 6e-10},
            "k_isc": 0.26e6}

    p_ex_array = np.ones(molecules_disposition.shape) * p_ex
    p_sted_array = np.ones(molecules_disposition.shape) * p_sted
    dwelltime_array = np.ones(molecules_disposition.shape) * dwelltime
    roi = 'max'

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(488e-9)
    laser_sted = base.DonutBeam(575e-9, zero_residual=0)
    detector = base.Detector(noise=noise, background=background)
    objective = base.Objective()
    fluo = base.Fluorescence(**egfp)
    datamap = base.Datamap(molecules_disposition, pixelsize)
    microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
    t0 = time.time()
    i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)
    dt_cache = time.time()-t0
    datamap.set_roi(i_ex, roi)

    t0 = time.time()
    acquisition, bleached, intensity = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, float(dwelltime), float(p_ex), float(p_sted),
                                                                    bleach=bleach, update=False)
    dt_get_signal = time.time()-t0

    data = {"Datamap roi":datamap.whole_datamap[datamap.roi], "Bleached datamap":bleached["base"][datamap.roi],
            "Acquired signal (photons)":acquisition, "dt cache":dt_cache, "dt get_signal":dt_get_signal,
            "pixelsize":pixelsize}
    return data
