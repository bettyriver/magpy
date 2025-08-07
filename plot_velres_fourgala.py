#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 18:25:38 2024

@author: ymai0110
"""

# read all four files and find the largest size


import sys
sys.path.insert(0,'/project/blobby3d/magpy/')
from postb3d_compmap_vel import make_vel_compmap,make_vel_compmap_combine
from os import listdir
parent_path_1 = "/project/blobby3d/Blobby3dYifan/v221/data_v221_rerun/"
parent_path_2 = "/project/blobby3d/Blobby3dYifan/v221/data_v221/"
fig_path = "/project/blobby3d/Blobby3dYifan/v221/talk/"


path = '/project/blobby3d/Blobby3dYifan/v221/data_v221_rerun/'

'''

magpiid_list = ['1202197197','1503208231','1204198199','1528197197']
i = 0
for magpiid in magpiid_list:
    if magpiid =='1202197197':
        datapath = parent_path_1+magpiid+'/'
    else:
        datapath = parent_path_2+magpiid+'/'
    if i==0:
        
        make_vel_compmap(datapath=datapath,figpath=fig_path,set_ticks=True,
                         set_title=magpiid,panel_pos='top')
    elif i<3:
        make_vel_compmap(datapath=datapath,figpath=fig_path,set_ticks=True,
                         set_title=magpiid,panel_pos='mid')
    else:
        make_vel_compmap(datapath=datapath,figpath=fig_path,set_ticks=True,
                         set_title=magpiid,panel_pos='bottom')


    i = i+1

'''


magpiid_list = ['1202197197','1503208231','1204198199','1528197197']
datapath_list = []
i = 0
for magpiid in magpiid_list:
    if magpiid =='1202197197':
        datapath = parent_path_1+magpiid+'/'
    else:
        datapath = parent_path_2+magpiid+'/'
    datapath_list.append(datapath)


    i = i+1


make_vel_compmap_combine(datapath_list,magpiid_list,figpath=fig_path,set_title='4res_combine',set_ticks=True)