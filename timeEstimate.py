#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:32:37 2022

@author: ymai0110
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import pandas as pd

def timepixels(pixels,a,b):
    time = a*pixels+b
    return time




pixels = np.array([1244,78,50,299])
times = np.array([164,4,3.5,27])

popt,pcov = curve_fit(timepixels,pixels,times)

plt.scatter(pixels,times)
plt.plot(pixels,timepixels(pixels,popt[0],popt[1]))
plt.show()


data = pd.read_csv("/Users/ymai0110/Documents/Blobby3D/sampleSelection/sampleSelection.csv")
data_pixel = data['sn3pixel']
data_select = data['select']

pixel_select = data_pixel[data_select==True]

time_select = timepixels(pixel_select, popt[0], popt[1])
plt.scatter(pixel_select, time_select)
plt.xlabel('pixels')
plt.ylabel('hours')
plt.show()