#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:01:55 2022

@author: ymai0110
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.io.fits as fits
import matplotlib as mpl

def moffat(x,y,a,b):
    f = (b-1)/np.pi/a/a*np.power((1+(x**2+y**2)/a**2),-b)
    return f

def two_gaussian(xdata_tuple,A1,A2,sigma1,sigma2):
    (x, y) = xdata_tuple
    f = A1*np.exp(-(x**2+y**2)/2/sigma1**2)+A2*np.exp(-(x**2+y**2)/2/sigma2**2)
    return f.ravel()

def three_gaussian(xdata_tuple,A1,A2,A3,sigma1,sigma2,sigma3):
    (x, y) = xdata_tuple
    f = A1*np.exp(-(x**2+y**2)/2/sigma1**2)+A2*np.exp(-(x**2+y**2)/2/sigma2**2)+A3*np.exp(-(x**2+y**2)/2/sigma3**2)
    return f.ravel()

def mof_to_gauss(alpha,beta,plot=False,magpiid=None):

    #beta = 42.77066906841987
    #alpha = 13.38382023870645
    
    x = np.linspace(-24, 24, 1000)
    y = np.linspace(-24, 24, 1000)
    xx, yy = np.meshgrid(x, y)
    
    #t0 = moffat(x, y, alpha, beta)
    t1 = moffat(xx,yy,alpha,beta)
    
    
    
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    ydata = t1.ravel()
    
    popt, pcov = curve_fit(f=two_gaussian,xdata=xdata,ydata=ydata,
                           bounds=([0,0,0,0],[np.inf,np.inf,np.inf,np.inf]))
    
    t2 = two_gaussian(xdata,popt[0],popt[1],popt[2],popt[3]).reshape(1000,1000)
    
    if plot==True:
        
        fig,ax = plt.subplots(1,3)
        
        max_v = np.max(t1)
        min_v = np.min(t1)
        
        im0=ax[0].imshow(t1,vmin=min_v, vmax=max_v)
        cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
        cb0.ax.locator_params(nbins=5)
        ax[0].set_title('moffat')
        
        
        
        im1=ax[1].imshow(t2,vmin=min_v, vmax=max_v)
        cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
        cb1.ax.locator_params(nbins=5)
        ax[1].set_title('two_gaussian')
        
        max_res = np.max(t1-t2)
        
        im2=ax[2].imshow(t1-t2,cmap='RdYlBu',vmin=-max_res, vmax=max_res)
        cb2 = plt.colorbar(im2,ax=ax[2],fraction=0.047)
        cb2.ax.locator_params(nbins=5)
        ax[2].set_title('residual')
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.8, 
                    hspace=0.4)
        plt.suptitle(str(magpiid))
        plt.show()
    
    
    max_flux = np.max(t1)
    rr = np.sqrt(xx**2 + yy**2)
    fwhm = 2*np.min(rr[t1 < max_flux/2])
    print(fwhm)
    print(popt)
    
    weight1 = popt[0]/(popt[0]+popt[1])
    weight2 = popt[1]/(popt[0]+popt[1])
    
    # 0.2 arcsec per pixel
    gau_fwhm1 = 2*np.sqrt(2*np.log(2))*np.abs(popt[2])*0.2
    gau_fwhm2 = 2*np.sqrt(2*np.log(2))*np.abs(popt[3])*0.2
    
    if (popt[0] < 0) | (popt[1] < 0):
        print('something went wrong!')
        return 0
    
    return weight1, weight2, gau_fwhm1, gau_fwhm2
    
def mof_to_gauss_threeGau(alpha,beta,plot=False,magpiid=None):

    #beta = 42.77066906841987
    #alpha = 13.38382023870645
    
    x = np.linspace(-24, 24, 1000)
    y = np.linspace(-24, 24, 1000)
    xx, yy = np.meshgrid(x, y)
    
    #t0 = moffat(x, y, alpha, beta)
    t1 = moffat(xx,yy,alpha,beta)
    
    
    
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    ydata = t1.ravel()
    
    popt, pcov = curve_fit(f=three_gaussian,xdata=xdata,ydata=ydata,
                           bounds=([0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
    
    t2 = three_gaussian(xdata,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]).reshape(1000,1000)
    
    if plot==True:
        
        fig,ax = plt.subplots(1,3)
        
        max_v = np.max(t1)
        min_v = np.min(t1)
        
        im0=ax[0].imshow(t1,vmin=min_v, vmax=max_v)
        cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
        cb0.ax.locator_params(nbins=5)
        ax[0].set_title('moffat')
        
        
        
        im1=ax[1].imshow(t2,vmin=min_v, vmax=max_v)
        cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
        cb1.ax.locator_params(nbins=5)
        ax[1].set_title('three_gaussian')
        
        max_res = np.max(t1-t2)
        
        im2=ax[2].imshow(t1-t2,cmap='RdYlBu',vmin=-max_res, vmax=max_res)
        cb2 = plt.colorbar(im2,ax=ax[2],fraction=0.047)
        cb2.ax.locator_params(nbins=5)
        ax[2].set_title('residual')
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.8, 
                    hspace=0.4)
        plt.suptitle(str(magpiid))
        plt.show()
    
    
    max_flux = np.max(t1)
    rr = np.sqrt(xx**2 + yy**2)
    fwhm = 2*np.min(rr[t1 < max_flux/2])
    print(fwhm)
    print(popt)
    
    # ! unfinished
    
    weight1 = popt[0]/(popt[0]+popt[1])
    weight2 = popt[1]/(popt[0]+popt[1])
    
    # 0.2 arcsec per pixel
    gau_fwhm1 = 2*np.sqrt(2*np.log(2))*np.abs(popt[2])*0.2
    gau_fwhm2 = 2*np.sqrt(2*np.log(2))*np.abs(popt[3])*0.2
    
    if (popt[0] < 0) | (popt[1] < 0):
        print('something went wrong!')
        return 0
    
    return weight1, weight2, gau_fwhm1, gau_fwhm2

def plot_compar(datapath,magpiid):
    magpifile = fits.open(datapath + "MAGPI"+str(magpiid)+"_minicube.fits")
    psfhdr = magpifile[4].header
    # note that the datacubes have mistake, alpha is beta , beta is alpha
    beta = psfhdr['MAGPI PSF ZBAND MOFFAT ALPHA']
    alpha = psfhdr['MAGPI PSF ZBAND MOFFAT BETA']
    
    weight1, weight2, fwhm1, fwhm2 = mof_to_gauss(alpha=alpha, beta=beta, plot=True,magpiid=magpiid)
    
def plot_compar_threeGau(datapath,magpiid):
    magpifile = fits.open(datapath + "MAGPI"+str(magpiid)+"_minicube.fits")
    psfhdr = magpifile[4].header
    # note that the datacubes have mistake, alpha is beta , beta is alpha
    beta = psfhdr['MAGPI PSF ZBAND MOFFAT ALPHA']
    alpha = psfhdr['MAGPI PSF ZBAND MOFFAT BETA']
    
    weight1, weight2, fwhm1, fwhm2 = mof_to_gauss_threeGau(alpha=alpha, beta=beta, plot=True,magpiid=magpiid)
    