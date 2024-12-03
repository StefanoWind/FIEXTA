'''
Example of application of LiDARGO to AWAKEN volumetric formatted data from Cornell Halo XR at site A1.

!This example requires a significant CPU time to complete!
'''
 
import os
cd=os.getcwd()
pwd=os.path.dirname(cd)
import sys
sys.path.append(pwd)
import LIDARGO_standardize as stand
import LIDARGO_statistics as stats
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
filename1=os.path.join(pwd,'data/Example2/sa1.lidar.z05.a2.20240824.085030.user5.nc')
filename2=os.path.join(pwd,'data/Example2/sa1.lidar.z05.b0.20240824.085030.user5.awaken.vol.nc')
source_config_stand=os.path.join(pwd,'config/config_examples_stand.xlsx')
source_config_stats=os.path.join(pwd,'config/config_examples_stats.xlsx')

#%% Main

#standardization
lproc = stand.LIDARGO(filename1,source_config_stand, verbose=True)
lproc.process_scan(filename1,replace=True,save_file=True)

#average
lproc = stats.LIDARGO(filename2,source_config_stats, verbose=True)
lproc.process_scan(filename2,replace=True,save_file=True)