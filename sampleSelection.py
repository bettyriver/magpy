#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:10:26 2022

@author: ymai0110
"""

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from os import walk
import pandas as pd

def snpixel(fitsfile,snlim):
    file = fits.open(fitsfile)
    ha_f = file['Ha_F'].data
    ha_err = file['Ha_FERR'].data
    ha_sn = ha_f/ha_err
    ha_sn_1d = ha_sn.flatten()
    npixel = sum(ha_sn_1d >= snlim)
    
    return npixel

def inclination(semimaj,semimin):
    '''in rad'''
    inc = np.arccos(semimin/semimaj)
    return inc





path = "/Users/ymai0110/Documents/Blobby3D/sample/"
filenames = next(walk(path), (None, None, []))[2]  # [] if no file

# s/n pixel (sn>=3)
# redshift halpha 6563 nii 6583 magpi 4700-9351, limit of redshift is 0.420(nii) or 0.424(ha)
# inclination (30 deg <= inc <=  60 deg) (pi/6 < inc < pi/3)


path2 = '/Users/ymai0110/Documents/Blobby3D/'

# 
def createsnpixelcsv():
    for filename in filenames:
        field = filename[:9]
        # go to the emission line map dir
        emidir = path2 + 'emissionLineMap/' + field + '_GIST_EmissionLine_Maps/'
        maplist = next(walk(emidir), (None, None, []))[2]
        magpiidlist = []
        sn3pixellist = []
        for emimap in maplist:
            if emimap[0]== '.':
                continue
            magpiid = emimap[5:15]
            sn3pixel = snpixel(emidir+emimap,3)
            magpiidlist.append(magpiid)
            sn3pixellist.append(sn3pixel)
        df = pd.DataFrame({'MAGPIID':magpiidlist,'sn3pixel':sn3pixellist})
        df.to_csv(path2+'sampleSelection/'+field+'_snpixel.csv')

# all info into same table

magpiid = []
redshift = []
z1prob = []
semimaj = []
semimin = []
inc = []
sn3pixel = []
select = []

for filename in filenames:
    field = filename[:9]
    table1 = pd.read_csv(path2+'sample/'+field+'_profoundsources.csv')
    table2 = pd.read_csv(path2+'sampleSelection/'+field+'_snpixel.csv')
    field_magpiid = np.array(table1['MAGPIID'].tolist())
    field_redshift = np.array(table1['Z'].tolist())
    field_redshiftprob = np.array(table1['Z1_PROB'].tolist())
    field_semimaj = np.array(table1['semimaj'].tolist())
    field_semimin = np.array(table1['semimin'].tolist())
    field_inc = inclination(semimaj=field_semimaj, semimin=field_semimin)
    field_sn3pixel = []
    table2_id = np.array(table2['MAGPIID'].tolist())
    table2_sn = np.array(table2['sn3pixel'].tolist())
    for idd in field_magpiid:
        index = np.where(idd==table2_id)[0]
        if len(index)==0:
            field_sn3pixel.append(0)
            continue
        index = index[0]
        field_sn3pixel.append(table2_sn[index])
    magpiid = np.concatenate((magpiid,field_magpiid))
    redshift = np.concatenate((redshift,field_redshift))
    z1prob = np.concatenate((z1prob,field_redshiftprob))
    semimaj = np.concatenate((semimaj,field_semimaj))
    semimin = np.concatenate(( semimin,field_semimin))
    inc = np.concatenate((inc,field_inc))
    sn3pixel = np.concatenate((sn3pixel,field_sn3pixel))

select = (redshift <= 0.42) & (inc >= 0) & (inc <= np.pi/3) & (sn3pixel>50)

magpiid = magpiid.astype(int)

df = pd.DataFrame({'MAGPIID':magpiid,
                   'Z':redshift,
                   'Z1_PROB':z1prob,
                   'semimaj':semimaj,
                   'semimin':semimin,
                   'inclination':inc,
                   'sn3pixel':sn3pixel,
                   'select':select})
df.to_csv(path2+'sampleSelection/sampleSelection.csv')


plt.scatter(redshift[select],sn3pixel[select])
plt.xlabel("z")
plt.ylabel("N pixel sn>3")
plt.ylim(0,1000)
plt.show()

plt.hist(inc[select]*180/np.pi)
plt.xlabel("inclination(degree)")
plt.ylabel("N")
plt.show()