#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:48:44 2023

measure Ha Re in KROSS galaxies
code use in Artemis


estimated coding time: 30 mins

@author: ymai0110
"""

import numpy as np
import pandas as pd

from astropy.cosmology import LambdaCDM
lcdm = LambdaCDM(70,0.3,0.7)
import sys
sys.path.insert(0,'/project/blobby3d/Blobby3D/pyblobby3d/src/pyblobby3d/')
from post_blobby3d import PostBlobby3D
from moments import SpectralModel
import matplotlib.pyplot as plt

##Create an elliptical distance array for deprojected radial profiles.
#size=the size of the array. Should be the same as an image input for uf.radial_profile
#centre=centre of the ellipse
#ellip=ellipticity of the ellipse=1-b/a
#pa=position angle of the ellipse starting along the positive x axis of the image
#Angle type: 'NTE' = North-Through-East (Default). 'WTN'=West-Through-North
def ellip_distarr(size,centre,ellip,pa,scale=None, angle_type='NTE'):
    y,x=np.indices(size)
    x=x-centre[0]
    y=y-centre[1]
    r=np.sqrt(x**2 + y**2)
    theta=np.zeros(size)
    theta[np.where((x>=0) & (y>=0))]=np.arcsin((y/r)[np.where((x>=0) & (y>=0))])
    theta[np.where((x>=0) & (y<0))]=2.0*np.pi+np.arcsin((y/r)[np.where((x>=0) & (y<0))])
    theta[np.where((x<0) & (y>=0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y>=0))])
    theta[np.where((x<0) & (y<0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y<0))])
    if angle_type=='NTE':
        theta=theta+np.pi/2.0
    scdistarr=np.nan_to_num(np.sqrt((((np.sin(theta-pa))**2)+(1-ellip)**-2*(np.cos(theta-pa))**2))*r) ##SHOULD BE (1-ellip)**-2 !!!!
    #if scale=None:
    return scdistarr


def get_x(x,y,target):
    '''
    Define linear interpolation for half-light radius analysis
    x=vector of x-values
    y=vector of y-values
    target=target y value for which to extract the x value
    '''
    ind=find_nearest_index(y,target)
    if y[ind]<target:
        x1=ind
        x2=ind+1
    else:
        x1=ind-1
        x2=ind
    m=float(y[x2]-y[x1])/float(x[x2]-x[x1])
    b=y[x1]-m*x[x1]
    
    if m==0:
        return ind
    
    return float(target-b)/float(m)



def create_dist_array(flux_array):
    lens=int(flux_array.shape[1])
    dist=np.array([[np.sqrt((x-float(lens/2))**2+(y-float(lens/2))**2) for x in range(lens)] for y in range(lens)])
    return dist


def curve_of_growth(image,centre=None,distarr=None,mask=None):
    '''
    if centre==None:
        centre=np.array(image.shape,dtype=float)/2
    if distarr==None:
        y,x=np.indices(image.shape)
        distarr=np.sqrt((x-centre[0])**2 + (y-centre[1])**2)
    '''
    if mask==None:
        mask=np.zeros(image.shape)# from ones -> zeros
    else:
        mask = mask.data
    
    rmax=int(np.max(distarr))
    r=np.linspace(0,rmax,rmax+1)
    #print(r)
    cog=np.zeros(len(r))
    for i in range(0,len(r)):
        cog[i]=np.nansum(image[np.where((mask==0) & (distarr<i))]) #from 1 --> 0
    #print(cog)
    cog[0]=0.0
    cog=cog/cog[-1]
    #print(cog,cog[-1])
    return r,cog

def find_nearest_index(array,target):
    dist=np.abs(array-target)
    #print('dist',dist)
    target_index=dist.tolist().index(np.nanmin(dist))
    return target_index

def pix_to_kpc(radius_in_pix,z,CD2=5.55555555555556e-05):
    '''
    author: yifan

    Parameters
    ----------
    radius_in_pix : float
        radius get from cont_r50_ind=get_x(r,cog,0.5)
    z : float
        redshift
    CD2 : float
        CD2 in fits

    Returns
    -------
    None.

    '''
    ang = radius_in_pix * CD2 # deg
    distance = lcdm.angular_diameter_distance(z).value # angular diameter distance
    radius_in_kpc = ang*np.pi/180*distance*1000
    return radius_in_kpc

def arcsec_to_kpc(rad_in_arcsec,z):
    from astropy.cosmology import LambdaCDM
    lcdm = LambdaCDM(70,0.3,0.7)
    distance = lcdm.angular_diameter_distance(z).value # angular diameter distance, Mpc/radian
    rad_in_kpc = rad_in_arcsec * distance * np.pi/(180*3600)*1000
    return rad_in_kpc


# read KROSS list
file_list = pd.read_csv("/scratch/blobby3d/KROSS/fileslist.csv")
folder_list = file_list['folder'].to_numpy()
filename_list = file_list['filename'].to_numpy()

mat_csv = pd.read_csv("/scratch/blobby3d/KROSS/KROSScombined_data.csv")
mat_filename_list = mat_csv['Unnamed: 0'].to_numpy()

save_path = '/scratch/blobby3d/KROSS/KROSS_ReHa_b3d/'
parent_path_1 = '/scratch/blobby3d/KROSS/KROSS_Stott_ResultsMv/'
parent_path_2 = '/scratch/blobby3d/KROSS/KROSS_Stott_Reduction/'
sample=0




for filename in mat_filename_list:
    print(filename)
    folder = folder_list[filename_list==filename][0]
    print(folder)
    datapath_1 = parent_path_1+str(folder)+'/'+filename+'/'
    datapath_2 = parent_path_2+str(folder)+'/'+filename+'/'
    
    try:
    
        post_b3d = PostBlobby3D(
                samples_path=datapath_1+'posterior_sample.txt',
                data_path=datapath_2+'data.txt',
                var_path=datapath_2+'var.txt',
                metadata_path=datapath_2+'metadata.txt',
                nlines=2)
    except:
        print(filename+' file not exist')
        continue
    
    query = mat_filename_list==filename
    ang = mat_csv['PA_IM'][query].to_numpy()[0]
    b_a = mat_csv['B_O_A'][query].to_numpy()[0]
    ellip = 1 - b_a
    z = mat_csv['Z'][query].to_numpy()[0]
    
    haflux = post_b3d.maps[sample, 0]
    ## find xcen ycen
    ny, nx = haflux.shape
    y_list = np.array(list(range(ny)))
    x_list = np.array(list(range(nx)))
    
    flux_median = np.median(np.log10(haflux[haflux>0]))
    flux_std = np.std(np.log10(haflux[haflux>0]))

    flux_mask = np.log10(haflux) > flux_median+3*flux_std

    haflux[flux_mask] = np.nan

    x_cen = np.nansum(haflux*x_list[None,:])/np.nansum(haflux)
    y_cen = np.nansum(haflux*y_list[:,None])/np.nansum(haflux)

    #plot
    yy, xx = np.indices(haflux.shape)
    fig, ax = plt.subplots()
    plt.scatter(xx,yy,c=np.log10(haflux))#,vmax = -18, vmin=-20
    cb = plt.colorbar()
    cb.set_label('log ha flux')
    circle1 = plt.Circle((x_cen, y_cen), 0.5, color='r')
    ax.add_patch(circle1)
    plt.title(filename)
    plt.savefig(save_path+filename+'_hamap.png')
    plt.show() 
    dist_arr = ellip_distarr(size=haflux.shape,centre=(x_cen,y_cen),ellip=ellip,pa=ang)# angel_type = 'NTE' for KROSS data
    
    pa=ang
    ## mask here comes from minicube
    r,cog=curve_of_growth(haflux,distarr=dist_arr,mask=None)
    cont_r50_ind=get_x(r,cog,0.5)
    r50_kpc = pix_to_kpc(cont_r50_ind,z,2.77778E-05 )
    plt.plot(r,cog)
    plt.xlabel('radius')
    plt.ylabel('light_within_that_radius/total_flux')
    plt.vlines(x=cont_r50_ind,ymin=0,ymax=cog.max(),color='k')
    plt.title(filename)
    plt.savefig(save_path+filename+'_haprofile.png')
    plt.show()
    name_list = []
    r50_pix_list = []
    r50_kpc_list = []
    name_list.append(filename)
    r50_pix_list.append(cont_r50_ind)
    r50_kpc_list.append(r50_kpc)
    
    df = pd.DataFrame({'filename':name_list,
                       'r50_pix':r50_pix_list,
                       'r50_kpc':r50_kpc_list})
    df.to_csv(save_path+filename+'_ha_r50_di.csv')
    print(filename+' done.')
