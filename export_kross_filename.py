#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:43:28 2023

@author: ymai0110
"""

from os import listdir 
from os.path import isfile, join, isdir
import pandas as pd

mypath = '/scratch/blobby3d/KROSS/KROSS_Stott_ResultsMv/'
folder_list = list(range(8))

all_folders = []
all_files = []

for folder in folder_list:
    onlyfiles = [f for f in listdir(mypath+str(folder)+'/') if isdir(join(mypath+str(folder)+'/', f))]
    onlyfolder = [folder for f in onlyfiles]
    all_folders += onlyfolder
    all_files += onlyfiles    

df = pd.DataFrame({'folder':all_folders,
                   'filename':all_files})
df.to_csv('/scratch/blobby3d/KROSS/fileslist.csv')