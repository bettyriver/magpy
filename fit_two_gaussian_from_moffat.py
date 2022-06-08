#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:01:55 2022

@author: ymai0110
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def moffat(x,y,a,b):
    f = (b-1)/np.pi/a/a*np.power((1+(x**2+y**2)/a**2),-b)
    return f

def two_gaussian(xdata_tuple,A1,A2,sigma1,sigma2):
    (x, y) = xdata_tuple
    f = A1*np.exp(-(x**2+y**2)/2/sigma1**2)+A2*np.exp(-(x**2+y**2)/2/sigma2**2)
    return f.ravel()

def mof_to_gauss(alpha,beta,plot=False):

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
        plt.imshow(t1)
        plt.colorbar()
        plt.title('moffat')
        plt.show()
        
        plt.imshow(t2)
        plt.colorbar()
        plt.title('two_gaussian')
        plt.show()
        
        plt.imshow(t1-t2)
        plt.colorbar()
        plt.title('residual')
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
    
    