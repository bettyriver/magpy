

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:26:38 2022

plot 1205 galaxy into one maps, one colorbar

@author: ymai0110
"""

from pyblobby3d.b3dcomp import setup_comparison_maps,map_limits,colorbar
from pyblobby3d.b3dplot import plot_map,plot_colorbar,cmap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import dnest4 as dn4
from pyblobby3d import PostBlobby3D
from pyblobby3d import SpectralModel



def make_compmap(datapath):
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
    ##precon_flux = post_b3d.maps[sample, 0]
    precon_flux = post_b3d.precon_cubes[sample].sum(axis=2)
    precon_vel = post_b3d.maps[sample, 2]
    precon_vdisp = post_b3d.maps[sample, 3]
    
    
    
    # convolved
    
    
    ##con_flux = fit_con[0]
    con_flux = post_b3d.con_cubes[sample].sum(axis=2)
    con_vel = fit_con[2]
    con_vdisp = fit_con[3]
    
    # data 
    
    ##data_flux = fit_data[0]
    data_flux = post_b3d.data.sum(axis=2)
    data_vel = fit_data[2]
    data_vdisp = fit_data[3]
    
    
    
    # mask all
    mask = np.isnan(con_vel)
    precon_flux[mask] = np.nan
    precon_vel[mask] = np.nan
    precon_vdisp[mask] = np.nan
    con_flux[mask] = np.nan
    con_vel[mask] = np.nan
    con_vdisp[mask] = np.nan
    data_flux[mask] = np.nan
    data_vel[mask] = np.nan
    data_vdisp[mask] = np.nan
    
    # residual
    res_flux = data_flux - con_flux
    res_vel = data_vel - con_vel
    res_vdisp = data_vdisp - con_vdisp
    
    
    fig,ax = setup_comparison_maps(comp_shape=(3,3), map_shape=(21,21), figsize=(15,20))
    flux_max = np.nanmax(con_flux)
    flux_min = np.nanmin(con_flux)
    vel_max = np.nanmax(con_vel)
    vel_min = np.nanmin(con_vel)
    if np.abs(vel_min)>vel_max:
        vel_max = np.abs(vel_min)
    else:
        vel_min = -vel_max
    
    vdisp_max = np.nanmax(con_vdisp)
    vdisp_min = np.nanmin(con_vdisp)
    
    res_flux_max = np.nanmax(res_flux)
    res_flux_min = np.nanmin(res_flux)
    if np.abs(res_flux_min)>res_flux_max:
        res_flux_max = np.abs(res_flux_min)
    else:
        res_flux_min = -res_flux_max
    
    res_vel_max = 15
    res_vel_min = -15
    
    res_vdisp_max = 15
    res_vdisp_min = -15
    
    pct = 90
    
    flux_ha_lim = map_limits(data=con_flux,pct=pct)
    vel_lim = map_limits(data=con_vel,pct=pct)
    vdisp_lim = map_limits(data=data_vdisp,pct=pct)
    print(flux_ha_lim)
    
    plot_map(ax[0][0],precon_flux,clim=flux_ha_lim,logscale=True,title='Pre-convolved',ylabel='flux')
    plot_map(ax[0][1],con_flux,clim=flux_ha_lim,logscale=True,title='convolved')
    plot_map(ax[0][2],data_flux,clim=flux_ha_lim,logscale=True,cbar_label='log(Flux(Ha))',title='data')
    plot_map(ax[1][0],precon_vel,clim=vel_lim,cmap=cmap.v,ylabel='velocity')
    plot_map(ax[1][1],con_vel,clim=vel_lim,cmap=cmap.v)
    plot_map(ax[1][2],data_vel,clim=vel_lim,cmap=cmap.v,cbar_label='v(km/s)')
    plot_map(ax[2][0],precon_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,ylabel='velocity dispersion')
    plot_map(ax[2][1],con_vdisp,clim=vdisp_lim,cmap=cmap.vdisp)
    plot_map(ax[2][2],data_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,cbar_label='$\sigma_v$(km/s)')
    
    plot_map(ax[0][4], res_flux,clim=[res_flux_min,res_flux_max],cbar_label='$\Delta$Flux',cmap=cmap.residuals,title='residual')
    plot_map(ax[1][4], res_vel,clim=[res_vel_min,res_vel_max],cbar_label='$\Delta$v(km/s)',cmap=cmap.residuals)
    plot_map(ax[2][4], res_vdisp,clim=[res_vdisp_min,res_vdisp_max],cbar_label='$\Delta \sigma_v$(km/s)',cmap=cmap.residuals)
    
    mpb0 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=flux_ha_lim[0],vmax=flux_ha_lim[1]))
    mpb1 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vel_lim[0],vmax=vel_lim[1]),cmap=cmap.v)
    mpb2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vdisp_lim[0],vmax=vdisp_lim[1]),cmap=cmap.vdisp)
    
    colorbar(mpb0,cax=ax[0][3],clim=flux_ha_lim,label='log(Flux(Ha))')
    colorbar(mpb1,cax=ax[1][3],clim=vel_lim,label='$v$(km/s)')
    colorbar(mpb2,cax=ax[2][3],clim=vdisp_lim,label='$\sigma_v$(km/s)')
    
    fwhm = 0.996
    circle = plt.Circle(
            (1.1*fwhm, 1.1*fwhm),
            radius=fwhm,
            fill=False, edgecolor='r', linewidth=1.0)
    ax[0][2].add_artist(circle)
    
    #plot_colorbar(ax[0][3])
    #plot_colorbar(ax[1][3])
    #plot_colorbar(ax[2][3])
    #plot_colorbar(ax[0][5])
    #plot_colorbar(ax[1][5])
    #plot_colorbar(ax[2][5])
    
    #plt.savefig(datapath+"compmap.png",dpi=300,bbox_inches='tight')
    return flux_ha_lim

datapath = "/Users/ymai0110/Documents/Blobby3D/data/1508218274/"
lim = make_compmap(datapath)
