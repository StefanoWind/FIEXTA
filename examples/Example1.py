'''
Example of application of LiDARGO to AWAKEN RHI data from sc1.lidar.z01 data channel and plot of snapshots and QC flags.
'''

import os
cd=os.getcwd()
pwd=os.path.dirname(cd)
import sys
sys.path.append(pwd)
import LIDARGO_standardize as stand
import LIDARGO_statistics as stats
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs

#paths
filename1=os.path.join(pwd,'data/Example1/sc1.lidar.z01.a0.20230830.064613.user4.nc')
filename2=os.path.join(pwd,'data/Example1/sc1.lidar.z01.b0.20230830.064613.user4.awaken.rhi.nc')
source_config_stand=os.path.join(pwd,'config/config_examples_stand.xlsx')
source_config_stats=os.path.join(pwd,'config/config_examples_stats.xlsx')

#graphics
xlim=[-3000,3000]#[m]
zlim=[0,1500]#[m]
max_ele=34#[deg]

#%% Main

#standardization
lproc = stand.LIDARGO(filename1,source_config_stand, verbose=True)
lproc.process_scan(filename1,replace=False,save_file=True)

#average
lproc = stats.LIDARGO(filename2,source_config_stats, verbose=True)
lproc.process_scan(filename2,replace=False,save_file=True)

#read standardized data
Data=xr.open_dataset(filename2)

#coordinates
time=Data['time'].values

X=Data['x'].mean(dim='scanID').values
Z=Data['z'].mean(dim='scanID').values

X_all=Data['x'].values
Y_all=Data['y'].values
Z_all=Data['z'].values

#wind speed
rws1=Data['wind_speed'].where(Data['qc_wind_speed']==0).where(Data['elevation']<max_ele).values
rws2=Data['wind_speed'].where(Data['qc_wind_speed']==0).where(Data['elevation']>180-max_ele).values

#other
scan_mode=Data.attrs['scan_mode']
qc=Data['qc_wind_speed'].values

#%% Plots
plt.close('all')

#all snapshots
for rep in range(len(time[0,:])):
    plt.figure(figsize=(18,8))
    
    pc=plt.pcolor(X,Z,np.abs(rws1[:,:,rep]),cmap='coolwarm',vmin=0,vmax=10)
    plt.pcolor(X,Z,np.abs(rws2[:,:,rep]),cmap='coolwarm',vmin=0,vmax=10)
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(r'$z$ [m]')
    ax=plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_xticks(np.arange(xlim[0],xlim[1]+1,250))
    ax.set_yticks(np.arange(zlim[0],zlim[1]+1,250))
    plt.grid()
    
    ax.set_box_aspect(np.diff(zlim)/np.diff(xlim))
    
    cb=plt.colorbar(label='QC $|u_{LOS}|$ [m s$^{-1}$]',orientation='horizontal')   
    plt.title('Rep #'+str(rep)+': '+str(time[0,rep])[:-10].replace('T',' ')+' - '+str(time[-1,rep])[:-10].replace('T',' '))

    fig_name=os.path.join(cd,'figures/Example1',os.path.basename(filename2).replace('.nc','{i:02d}'.format(i=rep)+'.png'))
    os.makedirs(os.path.dirname(fig_name),exist_ok=True)
    plt.savefig(fig_name)
    plt.close()
    
print('Snapshots were saved in '+(os.path.dirname(fig_name)))

#%QC insight
plt.figure(figsize=(18,10))
for qc_sel in range(12):
    ax=plt.subplot(4,3,qc_sel+1)  
    sel=~np.isnan(X_all+Y_all+qc)*(qc==qc_sel)
    if qc_sel==0:
        sc=ax.scatter(X_all[sel],Z_all[sel],s=5,c='g',alpha=0.1)
        plt.title('good')
    else:
        ax.scatter(X_all[sel],Z_all[sel],s=5,c='r',alpha=0.1)
        plt.title(Data['qc_wind_speed'].attrs['bit_{qc_sel}_description'.format(qc_sel=qc_sel)].replace('Value rejected due to ','')[:-1])
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.grid()
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(r'$z$ [m]')
    ax=plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_xticks(np.arange(xlim[0],xlim[1]+1,1000))
    ax.set_yticks(np.arange(zlim[0],zlim[1]+1,1000))

plt.tight_layout()