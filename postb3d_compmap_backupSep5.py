#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:26:38 2022

plot 1205 galaxy into one maps

@author: ymai0110
"""
import sys
sys.path.insert(0,'/project/blobby3d/Blobby3D/pyblobby3d/src/pyblobby3d')

from b3dcomp import setup_comparison_maps,map_limits,colorbar
from b3dplot import plot_map,plot_colorbar,cmap

import numpy as np
import matplotlib.pyplot as plt

import dnest4 as dn4
from post_blobby3d import PostBlobby3D
from moments import SpectralModel
import matplotlib as mpl


def make_compmap(datapath,figpath=None,flux_scale_factor=1,fwhm=None,plot_nii=False,
                 set_title=None):
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
    precon_flux_ha = post_b3d.maps[sample, 0]*flux_scale_factor
    ##precon_flux = post_b3d.precon_cubes[sample].sum(axis=2)
    
    precon_flux_nii = post_b3d.maps[sample,1]*flux_scale_factor
    precon_nii_ha = precon_flux_nii/precon_flux_ha
    
    precon_vel = post_b3d.maps[sample, 2]
    precon_vdisp = post_b3d.maps[sample, 3]
    
    
    
    # convolved
    
    
    con_flux_ha = fit_con[0]*flux_scale_factor
    ##con_flux = post_b3d.con_cubes[sample].sum(axis=2)
    con_flux_nii = fit_con[1]*flux_scale_factor
    con_nii_ha = con_flux_nii/con_flux_ha
    con_vel = fit_con[2]
    con_vdisp = fit_con[3]
    
    # data 
    
    data_flux_ha = fit_data[0]*flux_scale_factor
    ##data_flux = post_b3d.data.sum(axis=2)
    data_flux_nii = fit_data[1]*flux_scale_factor
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
    
    if set_title is not None:
        fig.suptitle(set_title,fontsize=16)
    
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
    
    pct = 90
    
    flux_ha_lim = map_limits(data=con_flux_ha,pct=pct)
    flux_nii_lim = map_limits(data=con_flux_nii,pct=pct)
    nii_ha_lim = map_limits(data=con_nii_ha,pct=pct)
    vel_lim = map_limits(data=con_vel,pct=pct)
    vdisp_lim = map_limits(data=data_vdisp,pct=pct)
    
    
    
    res_flux_ha_lim = map_limits(data=res_flux_ha,pct=pct,absolute=True)
    res_flux_nii_lim = map_limits(data=res_flux_nii,pct=pct,absolute=True)
    res_nii_ha_lim = map_limits(data=res_nii_ha,pct=pct,absolute=True)
    res_vel_lim = map_limits(data=res_vel,pct=pct,absolute=True)
    res_vdisp_lim = map_limits(data=res_vdisp,pct=pct,absolute=True)
    
    
    
    ii = 0
    
    if plot_nii == True:
        ii = 2
    
    
    plot_map(ax[0][0],precon_flux_ha,clim=flux_ha_lim,logscale=True,title='Pre-convolved',ylabel='flux_ha')
    plot_map(ax[0][1],con_flux_ha,clim=flux_ha_lim,logscale=True,title='convolved')
    plot_map(ax[0][2],data_flux_ha,clim=flux_ha_lim,logscale=True,cbar_label='log(Flux(Ha))',title='data')
    
    if plot_nii == True:
        plot_map(ax[1][0],precon_flux_nii,clim=flux_nii_lim,logscale=True,ylabel='flux_nii')
        plot_map(ax[1][1],con_flux_nii,clim=flux_nii_lim,logscale=True)
        plot_map(ax[1][2],data_flux_nii,clim=flux_nii_lim,logscale=True,cbar_label='log(Flux(nii))')
        plot_map(ax[1][4],res_flux_nii)
        
        plot_map(ax[2][0],precon_nii_ha,clim=nii_ha_lim,cmap='Oranges',ylabel='nii/ha')
        plot_map(ax[2][1],con_nii_ha,clim=nii_ha_lim,cmap='Oranges')
        plot_map(ax[2][2],data_nii_ha,clim=nii_ha_lim,cmap='Oranges')
        plot_map(ax[2][4],res_nii_ha,cmap=cmap.residuals)
    
    plot_map(ax[ii+1][0],precon_vel,clim=vel_lim,cmap=cmap.v,ylabel='velocity')
    plot_map(ax[ii+1][1],con_vel,clim=vel_lim,cmap=cmap.v)
    plot_map(ax[ii+1][2],data_vel,clim=vel_lim,cmap=cmap.v,cbar_label='v(km/s)')
    plot_map(ax[ii+2][0],precon_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,ylabel='velocity dispersion')
    plot_map(ax[ii+2][1],con_vdisp,clim=vdisp_lim,cmap=cmap.vdisp)
    plot_map(ax[ii+2][2],data_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,cbar_label='$\sigma_v$(km/s)')
    
    plot_map(ax[0][4], res_flux_ha,clim=res_flux_ha_lim,cbar_label='$\Delta$Flux',cmap=cmap.residuals,title='residual')
    if plot_nii == True:
        plot_map(ax[1][4], res_flux_nii,clim=res_flux_nii_lim,cmap=cmap.residuals)
        plot_map(ax[2][4], res_nii_ha,clim=res_nii_ha_lim,cmap=cmap.residuals)
    
    plot_map(ax[ii+1][4], res_vel,clim=res_vel_lim,cbar_label='$\Delta$v(km/s)',cmap=cmap.residuals)
    plot_map(ax[ii+2][4], res_vdisp,clim=res_vdisp_lim,cbar_label='$\Delta \sigma_v$(km/s)',cmap=cmap.residuals)
    
    mpb0 = mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=flux_ha_lim[0],vmax=flux_ha_lim[1]))
    mpb1 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vel_lim[0],vmax=vel_lim[1]),cmap=cmap.v)
    mpb2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vdisp_lim[0],vmax=vdisp_lim[1]),cmap=cmap.vdisp)
    
    colorbar(mpb0,cax=ax[0][3],clim=flux_ha_lim,label='Flux(Ha)')
    colorbar(mpb1,cax=ax[ii+1][3],clim=vel_lim,label='$v$(km/s)')
    colorbar(mpb2,cax=ax[ii+2][3],clim=vdisp_lim,label='$\sigma_v$(km/s)')
    
    if plot_nii == True:
        mpb3 = mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=flux_nii_lim[0],vmax=flux_nii_lim[1]))
        mpb4 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=nii_ha_lim[0],vmax=nii_ha_lim[1]),cmap='Oranges')
        colorbar(mpb3,cax=ax[1][3],clim=flux_nii_lim,label='Flux(NII)')
        colorbar(mpb4,cax=ax[2][3],clim=nii_ha_lim)
    
    # residual colorbar
    
    res_mpb0 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_flux_ha_lim[0],vmax=res_flux_ha_lim[1]),cmap=cmap.residuals)
    res_mpb1 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_vel_lim[0],vmax=res_vel_lim[1]),cmap=cmap.residuals)
    res_mpb2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_vdisp_lim[0],vmax=res_vdisp_lim[1]),cmap=cmap.residuals)
    res_mpb3 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_flux_nii_lim[0],vmax=res_flux_nii_lim[1]),cmap=cmap.residuals)
    res_mpb4 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_nii_ha_lim[0],vmax=res_nii_ha_lim[1]),cmap=cmap.residuals)
    
    colorbar(res_mpb0,cax=ax[0][5],clim=res_flux_ha_lim,label='$\Delta$ (Flux(Ha))')
    colorbar(res_mpb1,cax=ax[ii+1][5],clim=res_vel_lim,label='$\Delta v$(km/s)')
    colorbar(res_mpb2,cax=ax[ii+2][5],clim=res_vdisp_lim,label='$\Delta \sigma_v$(km/s)')
    if plot_nii == True:
        colorbar(res_mpb3,cax=ax[1][5],clim=res_flux_nii_lim,label='$\Delta$ (Flux(NII))')
        colorbar(res_mpb4,cax=ax[2][5],clim=res_nii_ha_lim)
    
    
    
    
    
    if fwhm is not None:
    
        circle = plt.Circle(
                (1.1*fwhm, 1.1*fwhm),
                radius=fwhm,
                fill=False, edgecolor='r', linewidth=1.0)
        ax[0][2].add_artist(circle)
    
    if figpath == None:
        figpath = datapath
    
    if set_title == None:
        plt.savefig(figpath+"compmap.pdf",dpi=300,bbox_inches='tight')
    else:
        plt.savefig(figpath+"compmap_"+set_title+".pdf",dpi=300,bbox_inches='tight')
