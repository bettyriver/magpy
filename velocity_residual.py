#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:57:23 2023



@author: ymai0110
"""


import numpy as np
import sys
sys.path.insert(0,'/Users/ymai0110/Documents/myPackages/pyblobby3d/')
from post_blobby3d import PostBlobby3D
from moments import SpectralModel

#from pyblobby3d import PostBlobby3D
#from pyblobby3d import SpectralModel
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt

def velocity_residual_hasn_weight(datapath,hasn_path,sample_num,sncut=5):
    '''calculate the ha_sn weighted average velocity residual'''
    
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    
    
    hasn_fits = fits.open(hasn_path)
    
    ha_sn_cut = hasn_fits[0].data
    
    
    
    # choose a sample
    sample = sample_num
    sm = SpectralModel(
            lines=[[6562.81], [6583.1, 6548.1, 0.3333]],
            lsf_fwhm=0.846,)
    wave = post_b3d.metadata.get_axis_array('r')
    fit_con, fit_var_con = sm.fit_cube(wave, post_b3d.con_cubes[sample], post_b3d.var)
    
    fit_data, fit_var_data = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
    
    # preconvolved
    precon_vel = post_b3d.maps[sample, 2]
    
    # convolved
    con_vel = fit_con[2]
    
    # data 
    data_vel = fit_data[2]
    
    
    # mask all
    mask = np.isnan(con_vel)
    precon_vel[mask] = np.nan
    con_vel[mask] = np.nan
    data_vel[mask] = np.nan
    #ha_sn[mask]  # can't do this, as my data has been cut....need to cut ha map
    
    # residual
    res_vel = data_vel - con_vel
    
    res_vel_abs = np.abs(res_vel)
    
    # vel and ha_sn_mask have nan value, need to use masked avearage
    
    #res_vel_abs_masked = np.ma.MaskedArray(res_vel_abs, mask=np.isnan(res_vel_abs))
    #average = np.ma.average(res_vel_abs_masked, weights = ha_sn_cut)
    
    ha_sn_cut[np.isnan(res_vel_abs)] = 0
    ha_sn_cut[np.isnan(ha_sn_cut)] = 0
    res_vel_abs[ha_sn_cut==0] = 0
    
    
    res_vel_abs[ha_sn_cut<sncut] = 0
    ha_sn_cut[ha_sn_cut<sncut] = 0
    
    
    average = np.average(res_vel_abs,weights=ha_sn_cut)
    
    
    
    
    return average, res_vel_abs, ha_sn_cut, post_b3d
    
    
    
def velocity_residual_median(datapath,hasn_path,sample_num,sncut):
    
    '''calculate the median of velocity residual'''
    
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    
    hasn_fits = fits.open(hasn_path)
    
    ha_sn_cut = hasn_fits[0].data
    
    # choose a sample
    sample = sample_num
    sm = SpectralModel(
            lines=[[6562.81], [6583.1, 6548.1, 0.3333]],
            lsf_fwhm=0.846,)
    wave = post_b3d.metadata.get_axis_array('r')
    fit_con, fit_var_con = sm.fit_cube(wave, post_b3d.con_cubes[sample], post_b3d.var)
    
    fit_data, fit_var_data = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
    
    # preconvolved
    precon_vel = post_b3d.maps[sample, 2]
    
    # convolved
    con_vel = fit_con[2]
    
    # data 
    data_vel = fit_data[2]
    
    
    # mask all
    mask = np.isnan(con_vel)
    precon_vel[mask] = np.nan
    con_vel[mask] = np.nan
    data_vel[mask] = np.nan
    #ha_sn[mask]  # can't do this, as my data has been cut....need to cut ha map
    
    # residual
    res_vel = data_vel - con_vel
    res_vel_abs = np.abs(res_vel)
    res_vel_abs[ha_sn_cut<sncut] = np.nan
    
    median = np.nanmedian(res_vel_abs)
    
    return median



def velocity_residual_hasn_and_meidna(post_b3d,hasn_path,sample_num,sncut=5):
    '''calculate the ha_sn weighted average velocity residual'''
    
    #post_b3d = PostBlobby3D(
    #        samples_path=datapath+'posterior_sample.txt',
    #        data_path=datapath+'data.txt',
    #        var_path=datapath+'var.txt',
    #        metadata_path=datapath+'metadata.txt',
    #        nlines=2)
    
    
    hasn_fits = fits.open(hasn_path)
    
    ha_sn_cut = hasn_fits[0].data
    
    
    
    # choose a sample
    sample = sample_num
    sm = SpectralModel(
            lines=[[6562.81], [6583.1, 6548.1, 0.3333]],
            lsf_fwhm=0.846,)
    wave = post_b3d.metadata.get_axis_array('r')
    fit_con, fit_var_con = sm.fit_cube(wave, post_b3d.con_cubes[sample], post_b3d.var)
    
    fit_data, fit_var_data = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
    
    # preconvolved
    precon_vel = post_b3d.maps[sample, 2]
    
    # convolved
    con_vel = fit_con[2]
    
    # data 
    data_vel = fit_data[2]
    
    
    # mask all
    mask = np.isnan(con_vel)
    precon_vel[mask] = np.nan
    con_vel[mask] = np.nan
    data_vel[mask] = np.nan
    #ha_sn[mask]  # can't do this, as my data has been cut....need to cut ha map
    
    # residual
    res_vel = data_vel - con_vel
    
    res_vel_abs = np.abs(res_vel)
    
    
    # calculate median before changing nan value to 0
    res_vel_abs[ha_sn_cut<sncut] = np.nan
    median = np.nanmedian(res_vel_abs)
    
    # 0 weight for nan data
    
    ha_sn_cut[np.isnan(res_vel_abs)] = 0
    ha_sn_cut[np.isnan(ha_sn_cut)] = 0
    res_vel_abs[ha_sn_cut==0] = 0
    
    if np.sum(ha_sn_cut)==0:
        return -999,-999
    
    average_hasn = np.average(res_vel_abs,weights=ha_sn_cut)
    
    
    return average_hasn, median

'''

# v2 code, calculate vel_res_ha_sn, vel_res_median, vmax, vel_res_ha/vmax,
# vel_res_median /vmax

parent_path = '/project/blobby3d/Blobby3dYifan/v221/data_v221/'
hafits_path = '/project/blobby3d/Blobby3dYifan/v221/ha_sn_cut_v221/'
magpi_csv = pd.read_csv('/project/blobby3d/Blobby3dYifan/v221/pbs_v221/vdisp_sfr_mass_re.csv')



magpiid = magpi_csv['MAGPIID'].to_numpy()
for idd in magpiid:
    ave_vel_hasn  = []
    median_vel = []
    
    id_str = str(idd)
    
    
    
    ave_vel_hasn.append(velocity_residual_hasn_weight(datapath=parent_path+id_str+'/',
                                                      hasn_path=hafits_path+id_str+'_ha_sn_cut.fits'))
    
    median_vel.append(velocity_residual_median(datapath=parent_path+id_str+'/'))
    
df = pd.DataFrame({'MAGPIID':magpiid,
                   'ave_vel_res_hasn':ave_vel_hasn,
                   'median_vel_res':median_vel})
df.to_csv('ave_vel_res_hasn.csv')

'''


'''
# v1 code
parent_path = '/project/blobby3d/Blobby3dYifan/v221/data_v221/'
hafits_path = '/project/blobby3d/Blobby3dYifan/v221/ha_sn_cut_v221/'
magpi_csv = pd.read_csv('/project/blobby3d/Blobby3dYifan/v221/pbs_v221/vdisp_sfr_mass_re.csv')

ave_vel_hasn  = []
median_vel = []

magpiid = magpi_csv['MAGPIID'].to_numpy()
for idd in magpiid:
    id_str = str(idd)
    ave_vel_hasn.append(velocity_residual_hasn_weight(datapath=parent_path+id_str+'/',
                                                      hasn_path=hafits_path+id_str+'_ha_sn_cut.fits'))
    
    median_vel.append(velocity_residual_median(datapath=parent_path+id_str+'/'))
    
df = pd.DataFrame({'MAGPIID':magpiid,
                   'ave_vel_res_hasn':ave_vel_hasn,
                   'median_vel_res':median_vel})
df.to_csv('ave_vel_res_hasn.csv')
'''
parent_path = '/Users/ymai0110/Documents/Blobby3D/v221/data_v221/'
hafits_path = '/Users/ymai0110/Documents/Blobby3D/v221/ha_sn_cut_v221/'
magpi_csv = pd.read_csv('/Users/ymai0110/Documents/Blobby3D/v221/pbs_v221/vdisp_sfr_mass_re.csv')
'''
ave_vel_hasn  = []
median_vel = []

magpiid = magpi_csv['MAGPIID'].to_numpy()
for idd in magpiid[:1]:
    id_str = str(idd)
    
    average, res_vel_abs, ha_sn_cut, post_b3d = velocity_residual_hasn_weight(datapath=parent_path+id_str+'/',
                                                      hasn_path=hafits_path+id_str+'_ha_sn_cut.fits')
    
    
    #ave_vel_hasn.append(velocity_residual_hasn_weight(datapath=parent_path+id_str+'/',
    #                                                  hasn_path=hafits_path+id_str+'_ha_sn_cut.fits'))
    
   # median_vel.append(velocity_residual_median(datapath=parent_path+id_str+'/'))

    plt.hist(post_b3d.global_param['VMAX'],bins=40)
    plt.xlabel('VMAX [km/s]')
    plt.ylabel('N')
    plt.title(id_str)
    plt.show()
    
'''
magpiid = magpi_csv['MAGPIID'].to_numpy()
for idd in magpiid[:1]:
    id_str = str(idd)
    ave_vel_hasn  = []
    median_vel = []
    
    datapath=parent_path+id_str+'/'
    hasn_path=hafits_path+id_str+'_ha_sn_cut.fits'
    
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    
    vmax_allsample = post_b3d.global_param['VMAX']
    
    for sample_num in range(len(vmax_allsample)):
        average_hasn_sample, median_sample = velocity_residual_hasn_and_meidna(post_b3d=post_b3d,
                                                                                hasn_path=hasn_path,
                                                                                sample_num=sample_num,sncut=5)
        
        ave_vel_hasn.append(average_hasn_sample)
        median_vel.append(median_sample)
        print('{} of {}'.format(sample_num, len(vmax_allsample)))

'''
df = pd.DataFrame({'MAGPIID':magpiid,
                   'ave_vel_res_hasn':ave_vel_hasn,
                   'median_vel_res':median_vel})
df.to_csv('/Users/ymai0110/Documents/Blobby3D/v221/pbs_v221/ave_vel_res_hasn_test.csv')

#for i in range(10):
#    plt.imshow(post_b3d.maps[i,0])
    
from matplotlib.colors import LogNorm

# ha flux
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


# velocity map
fig, ax = plt.subplots(3,3,figsize=(10,10))

for i in range(9):
    indx = int(i/3)
    indy = i%3
    
    im = ax[indx][indy].imshow(post_b3d.maps[i,2]-np.nanmedian(post_b3d.maps[i,2]),vmin=-15,vmax=15,cmap='RdYlBu_r')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('velocity map [km/s]',fontsize=15)
plt.title(id_str)
plt.show()
'''

