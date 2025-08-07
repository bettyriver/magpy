#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:11:57 2023

@author: ymai0110
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def velocity_profile_60(xdata, vc, beta, gamma, rt):
    x, y = xdata
    r = np.sqrt(x**2+y**2)
    theta=np.zeros(x.shape)
    theta[np.where((x>=0) & (y>=0))]=np.arcsin((y/r)[np.where((x>=0) & (y>=0))])
    theta[np.where((x>=0) & (y<0))]=2.0*np.pi+np.arcsin((y/r)[np.where((x>=0) & (y<0))])
    theta[np.where((x<0) & (y>=0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y>=0))])
    theta[np.where((x<0) & (y<0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y<0))])
    
    v_profile = vc*np.power(1+rt/r,beta)/np.power(1+np.power(rt/r,gamma),1/gamma)*np.sin(60*np.pi/180)*np.cos(theta)
    return v_profile

def velocity_profile_30(xdata, vc, beta, gamma, rt):
    x, y = xdata
    r = np.sqrt(x**2+y**2)
    theta=np.zeros(x.shape)
    theta[np.where((x>=0) & (y>=0))]=np.arcsin((y/r)[np.where((x>=0) & (y>=0))])
    theta[np.where((x>=0) & (y<0))]=2.0*np.pi+np.arcsin((y/r)[np.where((x>=0) & (y<0))])
    theta[np.where((x<0) & (y>=0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y>=0))])
    theta[np.where((x<0) & (y<0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y<0))])
    
    v_profile = vc*np.power(1+rt/r,beta)/np.power(1+np.power(rt/r,gamma),1/gamma)*np.sin(30*np.pi/180)*np.cos(theta)
    return v_profile

glo_para = pd.read_csv('/Users/ymai0110/Documents/Blobby3D/v221/data_v221_rerun/1202197197/1202197197_rerun_global_param.csv')


# use the first sample
vc = glo_para['VMAX'][0]
beta = glo_para['VBETA'][0]
gamma = glo_para['VGAMMA'][0]
rt = glo_para['VSLOPE'][0]
popt_45 = np.array([glo_para['VMAX'][0], glo_para['VBETA'][0], 
                    glo_para['VGAMMA'][0], glo_para['VSLOPE'][0]])
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
xx, yy = np.meshgrid(x,y)
x = xx
y = yy

r = np.sqrt(xx**2+yy**2)
theta=np.zeros((100,100))
theta[np.where((x>=0) & (y>=0))]=np.arcsin((y/r)[np.where((x>=0) & (y>=0))])
theta[np.where((x>=0) & (y<0))]=2.0*np.pi+np.arcsin((y/r)[np.where((x>=0) & (y<0))])
theta[np.where((x<0) & (y>=0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y>=0))])
theta[np.where((x<0) & (y<0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y<0))])

vmap_45 = vc*np.power(1+rt/r,beta)/np.power(1+np.power(rt/r,gamma),1/gamma)*np.sin(45*np.pi/180)*np.cos(theta)

plt.imshow(vmap_45,vmin=-200,vmax=200,cmap='RdYlBu_r')
plt.show()


vmap_60 = vc*np.power(1+rt/r,beta)/np.power(1+np.power(rt/r,gamma),1/gamma)*np.sin(60*np.pi/180)*np.cos(theta)
vmap_30 = vc*np.power(1+rt/r,beta)/np.power(1+np.power(rt/r,gamma),1/gamma)*np.sin(30*np.pi/180)*np.cos(theta)


fig, ax = plt.subplots(2,3,figsize=(10,6))
im = ax[0][0].imshow(vmap_45,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[0][0].set_title("i=45 deg")
ax[0][1].imshow(vmap_60,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[0][1].set_title("i=60 deg")
ax[0][2].imshow(vmap_60-vmap_45,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[0][2].set_title('residual between 45 and 60')


ax[1][0].imshow(vmap_45,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[1][0].set_title("i=45 deg")
ax[1][1].imshow(vmap_30,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[1][1].set_title("i=30 deg")
ax[1][2].imshow(vmap_30-vmap_45,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[1][2].set_title('residual between 45 and 30')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('velocity map [km/s]',fontsize=15)
plt.show()


## fit deg 45 by 60 deg model
popt_60, pcov = curve_fit(velocity_profile_60, (xx.ravel(),yy.ravel()), vmap_45.ravel())
vmap_build_60 = velocity_profile_60((xx.ravel(),yy.ravel()), popt_60[0], popt_60[1], popt_60[2], popt_60[3])

## fit deg 45 by 30 deg model
popt_30, pcov = curve_fit(velocity_profile_30, (xx.ravel(),yy.ravel()), vmap_45.ravel())
vmap_build_30 = velocity_profile_30((xx.ravel(),yy.ravel()), popt_30[0], popt_30[1], popt_30[2], popt_30[3])

vmap_build_60 = vmap_build_60.reshape(100,100)
vmap_build_30 = vmap_build_30.reshape(100,100)

fig, ax = plt.subplots(2,3,figsize=(10,6))
im = ax[0][0].imshow(vmap_45,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[0][0].set_title("i=45 deg")
ax[0][1].imshow(vmap_build_60,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[0][1].set_title("i=60 deg")
ax[0][2].imshow(vmap_build_60-vmap_45,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[0][2].set_title('residual between 45 and 60')


ax[1][0].imshow(vmap_45,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[1][0].set_title("i=45 deg")
ax[1][1].imshow(vmap_build_30,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[1][1].set_title("i=30 deg")
ax[1][2].imshow(vmap_build_30-vmap_45,vmin=-200,vmax=200,cmap='RdYlBu_r')
ax[1][2].set_title('residual between 45 and 30')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('velocity map [km/s]',fontsize=15)
plt.show()