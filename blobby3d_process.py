#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:12:49 2023

@author: ymai0110
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import cm, colors
from matplotlib.patches import Ellipse
from hyperfit.linfit import LinFit
import sys
sys.path.insert(0,'/Users/ymai0110/Documents/myPackages/pyblobby3d/')
from post_blobby3d import PostBlobby3D


def dist_to_SAMI_SFMS(logM,logSFR):
    # SFMS best fit from Renzini 2015
    a = 0.76
    b = -1
    c = -7.64
    distance = np.abs(a*logM+b*logSFR+c)/np.sqrt(a**2+b**2)
    
    y = 0.76*logM - 7.64
    if logSFR >= y:
        sign = 1
    else:
        sign = -1
    distance = distance * sign
    
    return distance

def arcsec_to_kpc(rad_in_arcsec,z):
    from astropy.cosmology import LambdaCDM
    lcdm = LambdaCDM(70,0.3,0.7)
    distance = lcdm.angular_diameter_distance(z).value # angular diameter distance, Mpc/radian
    rad_in_kpc = rad_in_arcsec * distance * np.pi/(180*3600)*1000
    return rad_in_kpc

def kpc_to_arcsec(rad_in_kpc,z):
    from astropy.cosmology import LambdaCDM
    lcdm = LambdaCDM(70,0.3,0.7)
    distance = lcdm.angular_diameter_distance(z).value # angular diameter distance, Mpc/radian
    rad_in_arcsec = rad_in_kpc / distance / np.pi*(180*3600)/1000
    return rad_in_arcsec

def dist_to_MAGPI_SFMS(logM,logSFR):
    # SFMS best fit from Marcie
    a = 0.73
    b = -1
    c = -7.482
    distance = np.abs(a*logM+b*logSFR+c)/np.sqrt(a**2+b**2)
    
    y = 0.73*logM - 7.482
    if logSFR >= y:
        sign = 1
    else:
        sign = -1
    distance = distance * sign
    
    return distance

def dist_to_MAGPI_SFMS_v2(logM,logSFR):
    '''also works for list'''
    # SFMS best fit from Marcie
    a = 0.73
    b = -1
    c = -7.482
    distance = np.abs(a*logM+b*logSFR+c)/np.sqrt(a**2+b**2)
    
    y = 0.73*logM - 7.482
    if type(logSFR)!= float:
        
        sign_array = np.zeros(len(logSFR))
        for i in range(len(logSFR)):
            if logSFR[i] >= y[i]:
                sign_array[i] = 1
            else:
                sign_array[i] = -1
            
        distance = distance*sign_array
        return distance
        
    else:
        if logSFR >= y:
            sign = 1
        else:
            sign = -1
        
        
        distance = distance * sign
    
        return distance

def dist_to_SAMI_SFMS_v2(logM,logSFR):
    '''also works for list'''
    # SFMS best fit from Renzini 2015
    a = 0.76
    b = -1
    c = -7.64
    distance = np.abs(a*logM+b*logSFR+c)/np.sqrt(a**2+b**2)
    
    y = 0.76*logM - 7.64
    if type(logSFR)!= float:
        
        sign_array = np.zeros(len(logSFR))
        for i in range(len(logSFR)):
            if logSFR[i] >= y[i]:
                sign_array[i] = 1
            else:
                sign_array[i] = -1
            
        distance = distance*sign_array
        return distance
        
    else:
        if logSFR >= y:
            sign = 1
        else:
            sign = -1
        
        
        distance = distance * sign
    
        return distance
    
    
def deltaMS_SAMI(logM,logSFR):
    logSFR_ms = 0.76*logM - 7.64
    deltaMS = logSFR - logSFR_ms
    
    return deltaMS

def deltaMS_MAGPI(logM, logSFR):
    logSFR_ms = 0.73*logM - 7.482
    deltaMS = logSFR - logSFR_ms
    
    return deltaMS

def deltaMS_KROSS(logM, logSFR):
    logSFR_ms = 0.62*logM - 5.3
    deltaMS = logSFR - logSFR_ms
    
    return deltaMS

def dist_to_KROSS_SFMS_v2(logM,logSFR):
    '''also works for list'''
    # SFMS best fit from marim 2011
    a = 0.62
    b = -1
    c = -5.3
    distance = np.abs(a*logM+b*logSFR+c)/np.sqrt(a**2+b**2)
    
    y = 0.62*logM - 5.3
    if type(logSFR)!= float:
        
        sign_array = np.zeros(len(logSFR))
        for i in range(len(logSFR)):
            if logSFR[i] >= y[i]:
                sign_array[i] = 1
            else:
                sign_array[i] = -1
            
        distance = distance*sign_array
        return distance
        
    else:
        if logSFR >= y:
            sign = 1
        else:
            sign = -1
        
        
        distance = distance * sign
    
        return distance



def sigmaSFR_vdisp_func(X, gradient, const, d):
    '''
    vdisp = gradient * SigmaSFR + const + d * flag
    
    MAGPI_flag = 1
    SAMI_flag = 0
    '''
    
    SigmaSFR, flag = X
    
    vdisp = gradient * SigmaSFR + const + d * flag
    
    return vdisp


def sigmaSFR_vdisp_func_onesample(logSigmaSFR, gradient, intercept):
    vdisp = gradient * logSigmaSFR + intercept
    
    return vdisp






def curve_fit_sigmaSFR_vdisp(func, logSigmaSFR, flag, logvdisp):
    '''
    

    Parameters
    ----------
    func : TYPE
        sigmaSFR_vdisp_func
    logSigmaSFR : TYPE
        DESCRIPTION.
    flag : TYPE
        DESCRIPTION.
    logvdisp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    popt, pcov = curve_fit(func, (logSigmaSFR, flag), logvdisp)
    gradient = popt[0]
    const = popt[1]
    d = popt[2]
    # one standard deviation errors of parameters
    perr = np.sqrt(np.diag(pcov))
    
    return gradient, const, d, perr




def compare_hyperfit_curvefit(data,cov,bounds,xlabel,ylabel):
    
    # hyper fit
    
    hf = LinFit(data, cov)

    # Run an MCMC
    #bounds = ((-10.0, 10.0), (0, 100), (1.0e-5, 500.0))
    mcmc_samples, mcmc_lnlike = hf.emcee(bounds, verbose=True)
    print(np.mean(mcmc_samples, axis=1), np.std(mcmc_samples, axis=1))
    
    sigmas = hf.get_sigmas()

    xvals = np.linspace(min(data[0]), max(data[0]), 1000)
    yvals = hf.coords[0] * xvals + hf.coords[1]

    # Generate ellipses
    ells = [
        Ellipse(
            xy=[data[0][i], data[1][i]],
            width=2.0 * np.sqrt(cov[0][0][i]),
            height=2.0 * np.sqrt(cov[1][1][i]),
        )
        for i in range(len(data[0]))
    ]

    # Make the plot
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 1.03, 0.83])
    for i, e in enumerate(ells):
        ax.add_artist(e)
        e.set_color(cm.viridis(sigmas[i] / np.amax(sigmas)))
        e.set_edgecolor("k")
        e.set_alpha(0.9)
    ax.plot(xvals, yvals, c="k", marker="None", ls="-", lw=1.3, alpha=0.9)
    ax.plot(xvals, yvals - hf.vert_scat, c="k", marker="None", ls="--", lw=1.3, alpha=0.9)
    ax.plot(xvals, yvals + hf.vert_scat, c="k", marker="None", ls="--", lw=1.3, alpha=0.9)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    #ax.set_xlim(-3, 2)
    #ax.set_ylim(0.9, 1.9)
    
    xlim = ax.get_xlim()
    
    ylim = ax.get_ylim()
    
    

    # Add the colourbar
    cb = fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=np.amax(sigmas)), cmap=cm.viridis),
        ax=ax,
        shrink=0.55,
        aspect=10,
        anchor=(-7.1, 0.95),
    )
    cb.set_label(label=r"$\sigma$", fontsize=14)
    plt.show()
    
    
    

    popt, pcov = curve_fit(sigmaSFR_vdisp_func_onesample, data[0], data[1])
    
    
    
    plt.errorbar(data[0],data[1],xerr=np.sqrt(cov[0][0]),yerr=np.sqrt(cov[1][1]),fmt='o')
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    x_arr = np.linspace(xlim[0],xlim[1],1000)
    y_arr_hyperfit = sigmaSFR_vdisp_func_onesample(x_arr, hf.coords[0],hf.coords[1])
    y_arr_curvefit = sigmaSFR_vdisp_func_onesample(x_arr,popt[0],popt[1])
    
    plt.plot(x_arr,y_arr_hyperfit,linestyle='--',label='hyperfit')
    plt.plot(x_arr,y_arr_curvefit,linestyle='-.',label='curvefit')
    plt.legend()
    
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    perr = np.sqrt(np.diag(pcov))
    
    
    
    print("hyperfit parameter: gradient = {:.4f}, intercept = {:.4f}".format(hf.coords[0],hf.coords[1]))
    print("curvefit parameter: gradient = {:.4f}+/-{:.4f}, intercept = {:.4f}+/-{:.4f}".format(popt[0],perr[0],popt[1],perr[1]))
    
    


    
def error_of_log10_data(data, data_err):
    err_log_data = data_err/(np.log(10)*data)
    
    return err_log_data



def plot_sample_flux(datapath,id_str):
    from matplotlib.colors import LogNorm
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    fig, ax = plt.subplots(3,3,figsize=(10,10))

    for i in range(9):
        indx = int(i/3)
        indy = i%3
        
        im = ax[indx][indy].imshow(post_b3d.maps[i,0],norm=LogNorm(vmin=1e-1,vmax=2000))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('ha flux',fontsize=15)
    plt.title(id_str)
    plt.show()

def plot_sample_velocitymap(datapath,id_str):
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    fig, ax = plt.subplots(3,3,figsize=(10,10))
    ax = ax.ravel()
    for i in range(9):
        indx = int(i/3)
        indy = i%3
        
        im = ax[i].imshow(post_b3d.maps[i,2]-np.nanmedian(post_b3d.maps[i,2]),vmin=-15,vmax=15,cmap='RdYlBu_r')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('velocity map [km/s]',fontsize=15)
    plt.title(id_str)
    plt.show()
