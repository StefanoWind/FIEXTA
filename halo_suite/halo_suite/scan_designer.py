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
from tkinter import Tk
from tkinter.filedialog import askopenfilename

root = Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

plt.close('all')

#%% Inputs
source=askopenfilename(
    title="Scan geometry file",
    filetypes=[("All files", "*.xlsx")],
    initialdir=cd,
)
assert os.path.isfile(source),  f'Invalid file "{source}"' 

mode=input('Scan mode (SSM or CSM): ')
reps=int(input('Repetitions: '))
volumetric=input('Volumetric (y/n): ')=='y'
reset=input('Reset (y/n): ')=='y'
name=input('Name: ')

if mode=='CSM':
    path_config=askopenfilename(
        title="Lidar configuration file",
        filetypes=[("All files", "*.yaml")],
        initialdir=cd,
    )
    assert os.path.isfile(path_config),  f'Invalid file "{path_config}"' 
    optimize=input('Optimize trajectory (y/n): ')=='y'
    
#%% Initialization
config=[]

#scan info
data=pd.read_excel(source)
azi=data['Azimuth [deg]'].values
ele=data['Elevation [deg]'].values

#configs
if mode=='SSM':
    scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=reps,identifier=name,volumetric=volumetric,reset=reset)
elif mode=='CSM':
    with open(path_config, 'r') as fid:
        config = yaml.safe_load(fid)    
        
    if optimize:
        ppr=int(input('PPR: '))
    else:
        ppr=0
        
    azi_dir=data['Azimuth scan direction'].values
    ele_dir=data['Elevation scan direction'].values
    azi_dir=azi_dir[~np.isnan(azi_dir)]
    ele_dir=ele_dir[~np.isnan(ele_dir)]
    
    scan_file_compiler(mode=mode,azi=azi,ele=ele,azi_dir=azi_dir,ele_dir=ele_dir,config=config,
                       ppr=ppr,optimize=optimize,identifier=name,repeats=reps,volumetric=volumetric,reset=reset)
    
