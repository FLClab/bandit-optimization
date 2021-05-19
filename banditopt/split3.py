# Version copied from the matlab code
import skimage.io as skio
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from pandas.plotting import table
import pandas as pd
import json

from skimage import filters

import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os



t0 = time.time()
#THERE IS A MAIN AT THE END OF THIS FILE
#------------------ Default Input variables ----------------
params_dict ={
    #Parameter in option in the matlab code
#    "Tg" : 6, #% 'First frame to sum:'
    "Nb_to_sum" : 1, #  The Tg infered from this variable override Tg
    "smooth_factor" : 0.2, #% 'Smoothing factor:'
    "im_smooth_cycles" : 0, #% 'Smoothing cycles image:'
    "phasor_smooth_cycles" : 1, #% 'Smoothing cycles phasor:'
    "foreground_threshold" : 15, #% 'Threshold value:'
    "tau_exc" : np.inf, #% 'Tau_exc'
    "intercept_at_origin" : False, #% 'Fix Intercept at origin'

    #Parameters that are calculated in th matlab code but that could be modified manually
    "M0" : None,
    "Min" : None,

    #Paramaters that are fixed in the matlab code
    "m0" : 1,
    "harm1" : 1, #MATLAB code: harm1=1+2*(h1-1), where h1=1
    "klog" : 4,
    }
#-----------------------------------------------------------




def SPLIT_smooth(im, sm, n):
    filt = 1/(8+1/sm)*np.array([[1,   1,  1],
                                [1, 1/sm, 1],
                                [1,   1,  1],])
    for _ in range(n):
        im = signal.convolve2d(im, filt, mode='same')
    return im

def mean_I_func(cte, tau_exc, jm1, k, nm1):
    #TODO: Pourquoi il y a une constante au début? C'est pas ce que la théorie indique...
    #TODO: Pourquoi pas diviser k/2, comme dans la formule (6) du 2ième paper?
    return cte * np.exp(-jm1/tau_exc) / (1 + k/2*jm1/nm1)
def normalized_psf(t, r, tau_exc, k, N, wc):
    T=N-1
    return np.exp(-(1 + (k/T)*t/2)*2*r**2/wc**2)


def split_sted(sted_stack_fname, params_dict, Nb_to_sum, smooth_factor, im_smooth_cycles, phasor_smooth_cycles, foreground_threshold, tau_exc,intercept_at_origin, m0, M0, Min, harm1, klog, show_plots=False, new_tau_exc=None, return_analysis_fig=False):

    if new_tau_exc:
        tau_exc = new_tau_exc


    if type(sted_stack_fname)==str:
        sted_stack = skio.imread(sted_stack_fname)
        sted_stack = np.moveaxis(sted_stack, 0, -1)
    else:
        sted_stack = sted_stack_fname

    X, Y, N = sted_stack.shape

    if Nb_to_sum:
        Tg = N-Nb_to_sum+1




    #TODO: I would have to handle division by zero for real images


    g = (sted_stack*np.cos(2*np.pi*(harm1)*np.arange(N)/N)).sum(axis=2)/sted_stack.sum(axis=2)
    g[np.isnan(g)] = 0
    s = (sted_stack*np.sin(2*np.pi*(harm1)*np.arange(N)/N)).sum(axis=2)/sted_stack.sum(axis=2)
    s[np.isnan(s)] = 0
    I_exc = np.exp(-np.arange(N)/tau_exc)
    g_exc = (I_exc*np.cos(2*np.pi*(harm1)*np.arange(N)/N)).sum()/I_exc.sum()
    s_exc = (I_exc*np.sin(2*np.pi*(harm1)*np.arange(N)/N)).sum()/I_exc.sum()

    g = g - g_exc
    s = s - s_exc
    g_smoothed = g[:,:]
    s_smoothed = s[:,:]
    g = SPLIT_smooth(g,smooth_factor,phasor_smooth_cycles)
    s = SPLIT_smooth(s,smooth_factor,phasor_smooth_cycles)
    g_smoothed = SPLIT_smooth(g_smoothed,smooth_factor,phasor_smooth_cycles+1)
    s_smoothed = SPLIT_smooth(s_smoothed,smooth_factor,phasor_smooth_cycles+1)

    modules = np.sqrt(s**2+g**2)
    modules_smoothed = np.sqrt(s**2+g**2)
    ##TODO: I would like to rename Ngat. What is the meaning, exactly?
    Ngat = sted_stack[:,:,Tg-1:N].sum(axis=2)

    # TODO: VERIFY FIX
    if len(np.unique(Ngat)) > 0:
        foreground_threshold = filters.threshold_otsu(Ngat)
    else:
        foreground_threshold = -1

    modules_smoothed[Ngat < foreground_threshold]=0
    #TODO: modimg, ne?
    x = g_smoothed[np.logical_and(Ngat > foreground_threshold, modules_smoothed > 0)]
    y = s_smoothed[np.logical_and(Ngat > foreground_threshold, modules_smoothed > 0)]
    if intercept_at_origin:
        slope = curve_fit(lambda x,slope : slope*x, x, y)[0][0]
        bias = 0
    else:
        slope, bias = tuple(np.polyfit(x, y, 1)) #In matlab they give starting points (for speed?)
    phi = np.arctan(slope)
#    print("slope:", round(slope, 4))
#    print("phi:", round(phi, 4))
    #TODO: I get slight differences of angle value for phi and g_smoothed, s_smoothed compared to matlab



    phases = np.arctan(s/g) #Why not removing the bias to calculate the modules too?
    s = s - (Ngat>foreground_threshold)*bias
    modules = np.sqrt(s**2+g**2)
    g_rotated = modules*np.cos(phases-phi)
    s_rotated = modules*np.sin(phases-phi)
    delta_m = np.std(g_rotated[Ngat>foreground_threshold])
    delta_n = np.std(s_rotated[Ngat>foreground_threshold])



    P = np.dstack([g,s])
    if Min is None:
        count, borders = np.histogram(modules[Ngat>foreground_threshold], bins=10)

        centers =  (borders[1:] + borders[:-1])/2
        idx = np.argwhere(count>0.2*count.max())[0][0]
        if idx==0:
            Min=0;
        else:
            Min=centers[idx]





    Pin= np.array([Min*np.cos(phi), Min*np.sin(phi)])
    if not M0:
        M0 = m0/2 * delta_n/abs(delta_m-delta_n)
    Pout = np.array([M0*np.cos(phi), M0*np.sin(phi)])
    print("Pout",np.round(Pout,4))
    print("Pin",np.round(Pin,4))


    fout = ((P-Pin)*Pout).sum(axis=2)/ ((Pout)**2).sum()
    fout = 1 / (1+np.exp(-klog*(fout-0.5)))
    fin = 1-fout

    Fin = fin*SPLIT_smooth(Ngat,smooth_factor,im_smooth_cycles)


#    print(time.time()-t0,"secondes")
    #------------ PLOTTING ---------------------

    size_im = 4
    nb_ims = (5, 3)
    phasor_analysis_fig, axes = plt.subplots(nb_ims[1],nb_ims[0],figsize=(nb_ims[0]*(size_im),nb_ims[1]*(size_im-1)) )
    axes = axes.flatten()
    i_plot = 0


    plt.sca(axes[i_plot])
    table(axes[i_plot], pd.DataFrame([params_dict]).T, loc=9, colWidths=[0.5])
    plt.axis('off')
    i_plot+=1




    plt.sca(axes[i_plot]); i_plot+=1
    plt.title('F1 (first image)')
    plt.imshow(sted_stack[:,:,0], cmap="hot")

    plt.sca(axes[i_plot]); i_plot+=1
    plt.title('F'+str(sted_stack.shape[2])+' (last image)')
    plt.imshow(sted_stack[:,:,-1], cmap="hot")

    plt.sca(axes[i_plot]); i_plot+=1
    plt.title('Ngat=sum(stack[Tg to N])')
    plt.imshow(Ngat, cmap="hot")

    plt.sca(axes[i_plot]); i_plot+=1
    plt.title('foreground_threshold = '+str(foreground_threshold))
    foreground = np.zeros(Ngat.shape)
    foreground[Ngat>foreground_threshold] = 1
    plt.imshow(foreground, cmap="hot")


#    plt.sca(axes[i_plot]); i_plot+=1
#
#    plt.title("Phasors ("+str(phasor_smooth_cycles)+" smooth cycle)")
#    plt.scatter(g, s, s=0.1)
#    plt.xlim(-1, 1)
#    plt.ylim(-0.5, 0.5)
#    plt.grid('on')
#    #plt.plot()
#
#    plt.sca(axes[i_plot]); i_plot+=1
#
#    plt.title("("+str(phasor_smooth_cycles+1)+" smooth cycle, filtered)")
#    plt.scatter(x, y, s=0.1)
#    x_domain = np.linspace(x.min(), x.max())
#    plt.plot(x_domain, x_domain*slope+bias, label='trendline (zerobias='+str(intercept_at_origin)+')')
#    plt.scatter(Pin[0], Pin[1] ,label='Pin')
#    plt.scatter(Pout[0], Pout[1], label='Pout')
#    plt.legend()
##    plt.xlim(0, 1)
##    plt.ylim(0, 0.5)
#
    plt.sca(axes[i_plot]); i_plot+=1
    plt.title('SPLIT image (Fin=fin*Ngat)')
    plt.imshow(Fin, cmap="hot")

    plt.sca(axes[i_plot]); i_plot+=1
    plt.title('Fout=(1-fin)*Ngat)')
    plt.imshow((1-fin)*Ngat, cmap="hot")
#
##    plt.sca(axes[i_plot]); i_plot+=1
##    matlab_split=skio.imread('split_matlab.png')
##    MSE = ((matlab_split/matlab_split.max() - Fin/Fin.max())**2).max()
##    plt.title('matlab: MaxSE='+str(round(MSE,8)) )
##    plt.imshow(matlab_split, cmap="hot")
#
    plt.sca(axes[i_plot]); i_plot+=1
    mean_I_stack = sted_stack.sum(axis=(0,1))/sted_stack.sum(axis=(0,1))[0]
    popt, pcov = curve_fit(lambda jm1, k, cte : mean_I_func(cte, tau_exc, jm1, k, N-1), np.arange(N), mean_I_stack)
    k, cte = popt[0], popt[1]
    plt.scatter(np.arange(N), mean_I_stack, )
    plt.plot(np.linspace(0,N-1,100),  mean_I_func(cte, tau_exc, np.linspace(0,N-1,100), k, nm1=N-1), label="k="+str(round(k, 4)))
    plt.legend()
#
#
    plt.sca(axes[i_plot]); i_plot+=1
    wc = 200e-9
    sigma_conf = wc/4
    r_values = []
    g_psf = []
    s_psf = []
    for i in range(13):
        r = i*sigma_conf/2
        r_values.append(r)
        I_psf = normalized_psf(np.arange(N), r, tau_exc, k, N, wc)
        g_psf.append((I_psf*np.cos(2*np.pi*(harm1)*np.arange(N)/N)).sum()/I_psf.sum())
        s_psf.append((I_psf*np.sin(2*np.pi*(harm1)*np.arange(N)/N)).sum()/I_psf.sum())
    plt.scatter(g_psf, s_psf)


    plt.scatter(x, y, s=0.1)
    x_domain = np.linspace(x.min(), x.max())
    plt.plot(x_domain, x_domain*slope+bias, label='trendline (zerobias='+str(intercept_at_origin)+')')
    plt.scatter(Pin[0], Pin[1] ,label='Pin')
    plt.scatter(Pout[0]+Pin[0], Pout[1]+Pin[1], label='Pout')
    plt.legend()




    for i in range(i_plot, len(axes)):
        plt.sca(axes[i])
        plt.axis('off')

    # plt.savefig("split.pdf")
#    os.system("open split.pdf")

    if show_plots:
        plt.show()


    if return_analysis_fig:
        return Fin, phasor_analysis_fig
    else:
        return Fin

def split_sted_for_optim(sted_stack, tau_exc, show_plots=False, return_analysis_fig=True):
    #param_dict is a global variable defined at the top of this file
    split_image = split_sted(sted_stack, params_dict,  **params_dict, show_plots=show_plots, new_tau_exc=tau_exc,return_analysis_fig= return_analysis_fig)
    return split_image


def main():
    pd.DataFrame([params_dict]).to_csv('params.csv')
#    split_image = split_sted("stack.tif", params_dict,  **params_dict, show_plots=True )
#    split_image = split_sted(skio.imread("1_STED75.0_Exc15.tiff"), params_dict,  **params_dict, show_plots=True )
#    split_image = split_sted(skio.imread("smul_im1.tiff"), params_dict,  **params_dict, show_plots=True )
#    split_image = split_sted(skio.imread("smul_im1_sources20nm.tiff"), params_dict,  **params_dict, show_plots=True )
#    split_image = split_sted(skio.imread("smul_im1_sources20nm_Isatdiv.tiff"), params_dict,  **params_dict, show_plots=True )
#    split_image = split_sted(skio.imread("smul_im1_sources20nm_noise100.tiff"), params_dict,  **params_dict, show_plots=True )
    split_image = split_sted(skio.imread("26_STED10.0_Exc16.0.tiff"), params_dict,  **params_dict, show_plots=True )

#    split_image = split_sted(skio.imread("smul_im1_sources20nm_noise200_normalized.tiff"), params_dict,  **params_dict, show_plots=True )

if __name__ == "__main__":
    main()













#-------------------------------------------------------------------------------------------

#foreground_threshold = 10
#def mk_Threshold_Mask(im, threshold):
#    #TODO: Note that ther is an (unused) option to use multiple thesholds in the matlab code
#    thr_mask=np.zeros(im.shape);
#    thr_mask(im>thr)=1;
#return thr_mask
#
#foreground_msk = mk_Threshold_Mask(Ngat1, foreground_threshold)
