# -*- coding: utf-8 -*-
"""
Characterize kinematic of scanning head
"""
import os
cd=os.path.dirname(__file__)
from matplotlib import pyplot as plt
import numpy as np
import yaml
from halo_suite.utilities import scan_file_compiler

#%% Inputs
path_config=os.path.join(cd,'configs/config.yaml')

#%% Initialization
#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)      

#%% Main

#write file for sampling time test
scan_file_compiler(mode='SSM',azi=90,ele=90,repeats=config['repeats_test'])
print(f'Test stare file saved. Run it on the lidar with different PPRs.')
    

