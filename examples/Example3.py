# -*- coding: utf-8 -*-
'''
Example of application of LiDARGO to AWAKEN VAD data from sa1.lidar.z05 data channel.
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
from scipy.optimize import curve_fit
import matplotlib
import numpy as np

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs

#paths
filename1=os.path.join(pwd,'data/Example3/sa1.lidar.z05.vad.a2.20240824.085030.user5.nc')
filename2=os.path.join(pwd,'data/Example3/sa1.lidar.z05.vad.b0.20240824.085030.user5.awaken.vad.nc')
source_config_stand=os.path.join(pwd,'config/config_examples_stand.xlsx')

min_N=5

#%% Functions
def radial_velocity(X,u,v,w):
    '''
    Radial velocity as a function of beam direction and wind vector
    
    Inputs:
    -----
    X:tuple
        azimuth and elevation vectors
    u: float
        W-E velocity
    v: float
        S-N velocity
    w: float
        vertical velocity
        
    Outputs:
    -----
    rws: vector of floats
        radial velocity
    
    '''
    azimuth,elevation=X
    
    rws=u*np.cos(np.radians(90-azimuth))*np.cos(np.radians(elevation))+\
        v*np.sin(np.radians(90-azimuth))*np.cos(np.radians(elevation))+\
        w*np.sin(np.radians(elevation))
        
    return rws

#%% Main

#standardization
lproc = stand.LIDARGO(filename1,source_config_stand, verbose=True)
lproc.process_scan(filename1,replace=True,save_file=True)

#read standardized data
Data=xr.open_dataset(filename2)

#average data
Data_avg=Data.mean(dim='scanID')
rws=Data_avg['wind_speed'].where(Data_avg['qc_wind_speed']==0).values
azimuth=Data_avg.azimuth.values
elevation=Data_avg.elevation.values
z=Data_avg.mean(dim='beamID').z.values

#wind retrieval
WS=z+np.nan
WD=z+np.nan
for i in range(len(Data_avg.range)):
    real=~np.isnan(rws[i,:])
    if np.sum(real)>min_N:
        U= curve_fit(radial_velocity, (azimuth[real], elevation[real]), rws[i,real])[0]
        WS[i]=(U[0]**2+U[1]**2)**0.5
        WD[i]=(270-np.degrees(np.arctan2(U[1],U[0])))%360
        
#%% Plots
plt.close('all')
plt.figure(figsize=(12,10))
plt.subplot(1,2,1)
plt.plot(WS,z,'.k',markersize=5)
plt.xlabel(r'Mean horizontal wind speed [m s $^{-1}$]')
plt.ylabel('Height [m a.g.l.]')
plt.grid()

plt.subplot(1,2,2)
plt.plot(WD,z,'.k',markersize=5)
plt.xlabel(r'Mean wind direction [$^\circ$]')
plt.ylabel('Height [m a.g.l.]')
plt.grid()
plt.tight_layout()
