'''
Example of application of LiDARGO to AWAKEN PPI raw data from UT Dallas WindCube 200S at site C1a and plot of QC flags.
'''

import os
cd=os.getcwd()
pwd=os.path.dirname(cd)
import sys
sys.path.append(pwd)
import LIDARGO_format as form
import LIDARGO_standardize as stand
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
filename0=os.path.join(pwd,'data/Example5/sc1.lidar.z02.00.20230822.071301.ppi.50.nc')
filename1=os.path.join(pwd,'data/Example5/sc1.lidar.z02.a0.20230822.071301.ppi.50.nc')
filename2=os.path.join(pwd,'data/Example5/sc1.lidar.z02.b0.20230822.071301.ppi.50.awaken.ppi.nc')
source_config_stand=os.path.join(pwd,'config/config_examples_stand.xlsx')

#formatting
model='WindCube 200S' #lidar model for formatting raw file
site='sc1' #site ID
z_id='02'#instrument ID (if multiple lidars are present)
data_level_format='a0' #data level after formatting

#graphics
xlim=[-6000,6000]#[m]
ylim=[0,6000]#[m]

#%% Main

#formatting
lproc = form.LIDARGO(verbose=True)
lproc.process_scan(filename0,model, site,z_id, data_level_format, replace=False,save_path=os.path.dirname(filename0), save_file=True)

#standardization
lproc = stand.LIDARGO(filename1,source_config_stand, verbose=True)
lproc.process_scan(filename1,replace=False,save_file=True)

#read standardized data
Data=xr.open_dataset(filename2)

#coordinates
X_all=Data['x'].values
Y_all=Data['y'].values

#other
qc=Data['qc_wind_speed'].values

#%% Plots
plt.close('all')

#%QC insight
plt.figure(figsize=(18,10))
for qc_sel in range(12):
    ax=plt.subplot(4,3,qc_sel+1)  
    sel=~np.isnan(X_all+Y_all+qc)*(qc==qc_sel)
    if qc_sel==0:
        sc=ax.scatter(X_all[sel],Y_all[sel],s=5,c='g',alpha=0.1)
        plt.title('good')
    else:
        ax.scatter(X_all[sel],Y_all[sel],s=5,c='r',alpha=0.1)
        plt.title(Data['qc_wind_speed'].attrs['bit_{qc_sel}_description'.format(qc_sel=qc_sel)].replace('Value rejected due to ','')[:-1])
    
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    plt.grid()
    ax=plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[0],xlim[1]+1,2000))
    ax.set_yticks(np.arange(ylim[0],ylim[1]+1,2000))

plt.tight_layout()