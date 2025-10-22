# -*- coding: utf-8 -*-
"""
Generate scan file from xlsx
"""

import os
cd=os.path.dirname(__file__)
from matplotlib import pyplot as plt
import yaml
from halo_suite.utilities import scan_file_compiler
import pandas as pd
import numpy as np
plt.close('all')

#%% Inputs
source=input('Scan file: ')
mode=input('Scan mode (SSM or CSM): ').lower()
reps=input('Repetitions: ')
volumetric=input('Volumetric (True or False): ')=='True'
path_config=input('Config file: ')
name=input('Name: ')

#%% Initialization

#scan info
data=pd.read_excel(source)
azi=data['Azimuth [deg]'].values
ele=data['Elevation [deg]'].values

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)    

if mode=='csm':
    azi_dir=data['Azimuth scan direction'].values
    ele_dir=data['Elevation scan direction'].values
    azi_dir=azi_dir[~np.isnan(azi_dir)]
    ele_dir=ele_dir[~np.isnan(ele_dir)]
    
scan_file_compiler(mode=mode,azi=azi,ele=ele,azi_dir=azi_dir,ele_dir=ele_dir,repeats=1,
                   identifier=name,volumetric=volumetric,config=config)