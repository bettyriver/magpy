#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:26:38 2022

plot 1205 galaxy into one maps

@author: ymai0110
"""

from b3dcomp import setup_comparison_maps
from pyblobby3d.b3dplot import plot_map,plot_colorbar,cmap

import numpy as np
import matplotlib.pyplot as plt

import dnest4 as dn4
from pyblobby3d import PostBlobby3D
from pyblobby3d import SpectralModel



def make_compmap(datapath,plot_nii=False):
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    
    # choose a sample
    sample = 0
    sm = SpectralModel(
            lines=[[6562.81], [6583.1, 6548.1, 0.3333]],
            lsf_fwhm=0.846,)
    wave = post_b3d.metadata.get_axis_array('r')
    fit_con, fit_err_con = sm.fit_cube(wave, post_b3d.con_cubes[sample], post_b3d.var)
    
    fit_data, fit_err_data = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
    
    
    
    # preconvolved
    # precon flux, velocity map, velocity dispersion map
    precon_flux_ha = post_b3d.maps[sample, 0]
    ##precon_flux = post_b3d.precon_cubes[sample].sum(axis=2)
    
    precon_flux_nii = post_b3d.maps[sample,1]
    precon_nii_ha = precon_flux_nii/precon_flux_ha
    
    precon_vel = post_b3d.maps[sample, 2]
    precon_vdisp = post_b3d.maps[sample, 3]
    
    
    
    # convolved
    
    
    con_flux_ha = fit_con[0]
    ##con_flux = post_b3d.con_cubes[sample].sum(axis=2)
    con_flux_nii = fit_con[1]
    con_nii_ha = con_flux_nii/con_flux_ha
    con_vel = fit_con[2]
    con_vdisp = fit_con[3]
    
    # data 
    
    data_flux_ha = fit_data[0]
    ##data_flux = post_b3d.data.sum(axis=2)
    data_flux_nii = fit_data[1]
    data_nii_ha = data_flux_nii/data_flux_ha
    data_vel = fit_data[2]
    data_vdisp = fit_data[3]
    
    
    
    # mask all
    mask = np.isnan(con_vel)
    precon_flux_ha[mask] = np.nan
    precon_flux_nii[mask] = np.nan
    precon_nii_ha[mask] = np.nan
    precon_vel[mask] = np.nan
    precon_vdisp[mask] = np.nan
    con_flux_ha[mask] = np.nan
    con_flux_nii[mask] = np.nan
    con_nii_ha[mask] = np.nan
    con_vel[mask] = np.nan
    con_vdisp[mask] = np.nan
    data_flux_ha[mask] = np.nan
    data_flux_nii[mask] = np.nan
    data_nii_ha[mask] = np.nan
    data_vel[mask] = np.nan
    data_vdisp[mask] = np.nan
    
    # residual
    res_flux_ha = data_flux_ha - con_flux_ha
    res_vel = data_vel - con_vel
    res_vdisp = data_vdisp - con_vdisp
    res_flux_nii = data_flux_nii - con_flux_nii
    res_nii_ha = data_nii_ha - con_nii_ha
    
    
    if plot_nii == True:
        fig,ax = setup_comparison_maps(comp_shape=(5,3), map_shape=(41,41), figsize=(15,20))
    else :
        fig,ax = setup_comparison_maps(comp_shape=(3,3), map_shape=(41,41), figsize=(15,20))
    
    flux_max = np.nanmax(con_flux_ha)+10
    flux_min = np.nanmin(con_flux_ha)
    vel_max = np.nanmax(con_vel)
    vel_min = np.nanmin(con_vel)
    if np.abs(vel_min)>vel_max:
        vel_max = np.abs(vel_min)
    else:
        vel_min = -vel_max
    
    vdisp_max = np.nanmax(con_vdisp)+20
    vdisp_min = np.nanmin(con_vdisp)
    nii_ha_max = np.nanmax(con_nii_ha)
    nii_ha_min = np.nanmin(con_nii_ha)
    
    res_flux_max = np.nanmax(res_flux_ha)
    res_flux_min = np.nanmin(res_flux_ha)
    if np.abs(res_flux_min)>res_flux_max:
        res_flux_max = np.abs(res_flux_min)
    else:
        res_flux_min = -res_flux_max
    
    res_vel_max = 15
    res_vel_min = -15
    
    res_vdisp_max = 20
    res_vdisp_min = -20
    
    ii = 0
    
    if plot_nii == True:
        ii = 2
    
    
    plot_map(ax[0][0],precon_flux_ha,colorbar=True,clim=[flux_min,flux_max],logscale=True,title='Pre-convolved',ylabel='flux_ha')
    plot_map(ax[0][1],con_flux_ha,colorbar=True,clim=[flux_min,flux_max],logscale=True,title='convolved')
    plot_map(ax[0][2],data_flux_ha,colorbar=True,clim=[flux_min,flux_max],logscale=True,cbar_label='log(Flux(Ha))',title='data')
    
    if plot_nii == True:
        plot_map(ax[1][0],precon_flux_nii,colorbar=True,clim=[flux_min,flux_max],logscale=True,ylabel='flux_nii')
        plot_map(ax[1][1],con_flux_nii,colorbar=True,clim=[flux_min,flux_max],logscale=True)
        plot_map(ax[1][2],data_flux_nii,colorbar=True,clim=[flux_min,flux_max],logscale=True,cbar_label='log(Flux(nii))')
        plot_map(ax[1][3],res_flux_nii,colorbar=True)
        
        plot_map(ax[2][0],precon_nii_ha,colorbar=True,clim=[nii_ha_min,nii_ha_max],ylabel='nii/ha')
        plot_map(ax[2][1],con_nii_ha,colorbar=True,clim=[nii_ha_min,nii_ha_max])
        plot_map(ax[2][2],data_nii_ha,colorbar=True,clim=[nii_ha_min,nii_ha_max])
        plot_map(ax[2][3],res_nii_ha,colorbar=True)
    
    plot_map(ax[ii+1][0],precon_vel,colorbar=True,clim=[vel_min,vel_max],cmap=cmap.v,ylabel='velocity')
    plot_map(ax[ii+1][1],con_vel,colorbar=True,clim=[vel_min,vel_max],cmap=cmap.v)
    plot_map(ax[ii+1][2],data_vel,colorbar=True,clim=[vel_min,vel_max],cmap=cmap.v,cbar_label='v(km/s)')
    plot_map(ax[ii+2][0],precon_vdisp,colorbar=True,clim=[vdisp_min,vdisp_max],cmap=cmap.vdisp,ylabel='velocity dispersion')
    plot_map(ax[ii+2][1],con_vdisp,colorbar=True,clim=[vdisp_min,vdisp_max],cmap=cmap.vdisp)
    plot_map(ax[ii+2][2],data_vdisp,colorbar=True,clim=[vdisp_min,vdisp_max],cmap=cmap.vdisp,cbar_label='$\sigma_v$(km/s)')
    
    plot_map(ax[0][3], res_flux_ha,colorbar=True,clim=[res_flux_min,res_flux_max],cbar_label='$\Delta$Flux',cmap=cmap.residuals,title='residual')
    plot_map(ax[ii+1][3], res_vel,colorbar=True,clim=[res_vel_min,res_vel_max],cbar_label='$\Delta$v(km/s)',cmap=cmap.residuals)
    plot_map(ax[ii+2][3], res_vdisp,colorbar=True,clim=[res_vdisp_min,res_vdisp_max],cbar_label='$\Delta \sigma_v$(km/s)',cmap=cmap.residuals)
    plt.savefig(datapath+"compmap.png",dpi=300,bbox_inches='tight')
