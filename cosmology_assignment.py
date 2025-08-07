#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:42:12 2022

@author: ymai0110
"""

import matplotlib.pyplot as plt 

import numpy as np
N = 300000
time_log = np.linspace(-44,22,N) 
time = np.power(10,time_log)
t_i_index = np.where(time_log>-36)[0][0] 
t_f_index = np.where(time_log>-34)[0][0] 
t_47000_index = np.where(time>47000*365.25*24*3600)[0][0] 
t_9Gyr_index = np.where(time>9*1e9*365.25*24*3600)[0][0]
t_i = time[t_i_index]
t_f = time[t_f_index]
t_47000 = time[t_47000_index] 
t_9Gyr = time[t_9Gyr_index]
scale_iii = np.zeros(N) 
scale_iv = np.zeros(N) 
scale_v = np.zeros(N) 
scale_vi = np.zeros(N)
factor = 1e-25
a_i = 5.63*1e-70
# radiation-inflation-radiation
scale_iii[0:t_i_index] = a_i*np.sqrt(time[0:t_i_index]/t_i)
scale_iii[t_i_index:t_f_index] = a_i*np.power(np.e,1e36*(time[t_i_index:t_f_index]-t_i)) 
scale_iii[t_f_index:] = a_i * np.power(np.e,1e36*(t_f-t_i)) * np.sqrt(time[t_f_index:]/t_f)
# horizon
scale_iv = 3*10**8*time*factor
# radiation-inflation-radiation-darkMatter
scale_v[0:t_47000_index] = scale_iii[0:t_47000_index]

a_47000 = scale_iii[t_47000_index]
scale_v[t_47000_index:] = a_47000*np.power(time[t_47000_index:]/t_47000,2/3)
# radiation-inflation-radiation-darkMatter-darkEnergy 
limcut = 10000
h0 = np.sqrt(2/3*1e-35)
scale_vi[:] = scale_v[:]
a_9Gyr = scale_v[t_9Gyr_index]
scale_vi[t_9Gyr_index:t_9Gyr_index+limcut] = a_9Gyr*np.power(np.e,h0*(time[t_9Gyr_index:t_9Gyr_index+limcut]-t_9Gyr)) 
scale_vi[t_9Gyr_index+limcut:]=np.nan
plt.plot(np.log10(time),np.log10(scale_iii),label='a(t)_iii') 
plt.plot(np.log10(time),np.log10(scale_iv),label='Horizon')
plt.axhline(y=1,c='k',linestyle='--') 
plt.axvline(x=np.log10(13.8*1e9*365.25*24*3600),c='k',linestyle='--') 
plt.xlabel('log(t),time')
plt.ylabel('log(a), scale factor')
plt.legend()
plt.title('radiation-inflation-radiation') 
plt.savefig('Q1iii.png',dpi=300,bbox_inches='tight') 
plt.show()
plt.plot(np.log10(time),np.log10(scale_iii),label='a(t)_iii') 
plt.plot(np.log10(time),np.log10(scale_iv),label='Horizon') 
plt.plot(np.log10(time),np.log10(scale_v),label='(v)') 
plt.axhline(y=1,c='k',linestyle='--') 
plt.axvline(x=np.log10(13.8*1e9*365.25*24*3600),c='k',linestyle='--') 
plt.legend()
plt.xlabel('log(t),time')
plt.ylabel('log(a), scale factor') 
plt.title('radiation-inflation-radiation-darkMatter') 
plt.savefig('Q1v.png',dpi=300,bbox_inches='tight') 
plt.show()
plt.plot(np.log10(time),np.log10(scale_iii),label='a(t)_iii') 
plt.plot(np.log10(time),np.log10(scale_iv),label='Horizon')

plt.plot(np.log10(time),np.log10(scale_v),label='a(t)_v') 
plt.plot(np.log10(time),np.log10(scale_vi),label='a(t)_vi') 
plt.axhline(y=1,c='k',linestyle='--') 
plt.axvline(x=np.log10(13.8*1e9*365.25*24*3600),c='k',linestyle='--') 
plt.xlabel('log(t),time')
plt.ylabel('log(a), scale factor')
plt.legend() 
plt.title('radiation-inflation-radiation-darkMatter-darkEnergy') 
plt.savefig('Q1vi.png',dpi=300,bbox_inches='tight')
plt.show()