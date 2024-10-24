# -*- coding: utf-8 -*-
'''
Example of application of LiDARGO to AWAKEN stare data from sa5.lidar.z03 data channel and wind spectra.
'''

import os
cd=os.getcwd()
pwd=os.path.dirname(cd)
import sys
sys.path.append(pwd)
import LIDARGO_standardize as stand
import xarray as xr
from matplotlib import pyplot as plt
import warnings
import matplotlib
import numpy as np
from scipy.signal import welch

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs

#paths
filename1=os.path.join(pwd,'data/Example4/sa5.lidar.z03.a0.20231009.205005.user5.nc')
filename2=os.path.join(pwd,'data/Example4/sa5.lidar.z03.b0.20231009.205005.user5.awaken.stare.nc')
source_config_stand=os.path.join(pwd,'config/config_examples_stand.xlsx')

welch_window=50#number of FFT point in Welch

#%% Main

#standardization
lproc = stand.LIDARGO(filename1,source_config_stand, verbose=True)
lproc.process_scan(filename1,replace=True,save_file=True)

#read standardized data
Data=xr.open_dataset(filename2)

rws=np.squeeze(Data['wind_speed'].where(Data['qc_wind_speed']==0).values)
fs=(np.float64(Data.time.diff(dim='scanID').mean('scanID').values)/10**9)**-1
r=Data.range.values

#calculate Welch power spectral density
psd=np.zeros((len(r),int(np.ceil(welch_window/2))))+np.nan
for i in range(len(r)):
    f, psd[i,:] = welch(rws[i,:]-np.nanmean(rws[i,:]), fs, nperseg=welch_window)
    
#%% Plots
plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

plt.figure(figsize=(18,8))
sel= np.where(np.sum(~np.isnan(psd),axis=0)>0)[0]
cmap = plt.cm.get_cmap('viridis')
colors = [cmap(i / (len(sel) - 1)) for i in range(len(sel))]
for s,c in zip(sel,colors):
    plt.loglog(f,psd[s,:],label=r'$r='+str(r[s])+'$m',color=c)

plt.xlabel(r'Frequency [Hz]')
plt.ylabel('Power spectral density [m$^2$ s$^{-1}$]')
plt.grid()
plt.legend()

