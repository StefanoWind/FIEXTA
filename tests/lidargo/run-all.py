"""
Run all examples
"""

import os
import pandas as pd
cd = os.getcwd()
pwd = os.path.dirname(cd)

import lidargo as lg
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import glob
import re

warnings.filterwarnings("ignore")
plt.close("all")

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.size"] = 14

# %% Inputs

# paths
source = os.path.join(pwd, "data/**/*nc")
source_config_stand = os.path.join(pwd, "config/config_examples_stand.xlsx")
source_config_stats = os.path.join(pwd, "config/config_examples_stats.xlsx")

#%% Initialization
files=glob.glob(source,recursive=True)
config_stand=pd.read_excel(source_config_stand).set_index('PARAMETER')

#%% Main
files_stand=[]
configs_stand=[]
for f in files:
    
    date_source=np.int64(re.search(r'\d{8}.\d{6}',f).group(0)[:8])
    
    #match standfardized config
    matches=[]
    for regex in config_stand.columns:
        match = re.findall(regex, f)
        sdate=config_stand[regex]['start_date']
        edate=config_stand[regex]['end_date']
        if len(match)>0 and date_source>=sdate and date_source<=edate:
            matches.append(regex)
            
    if len(matches)==1:
        files_stand.append(f)
        configs_stand.append(config_stand[matches[0]].to_dict())
            
        
        
for f,c in zip(files_stand,configs_stand):
    
    ## Instantiate a configuration object
    config = lg.LidarConfig(**c)

    # standardization
    lproc = lg.Standardize(f, config=config, verbose=True)
    lproc.process_scan(replace=True, save_file=True, make_figures=False)

