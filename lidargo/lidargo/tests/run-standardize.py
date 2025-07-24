# -*- coding: utf-8 -*-
"""
Standardize a list of files with associated configs.
"""

import re
import pandas as pd
import glob
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import lidargo as lg

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

#%% Inputs

# source = "../../data/lidargo/**/*a0*.nc"
# config_file = "../../configs/lidargo/config_examples_stand.xlsx"
# save_path='C:/Users/SLETIZIA/OneDrive - NREL/Desktop/241220_lidargo_test/examples'

# source = "C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/AWAKEN/LIDARGO_samples/data/propietary/awaken/**/*a0*nc"
# config_file = "C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/AWAKEN/LIDARGO_samples/config/config_awaken_b0.xlsx"
# save_path='C:/Users/SLETIZIA/OneDrive - NREL/Desktop/241220_lidargo_test/awaken'


source = "C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/AWAKEN/LIDARGO_samples/data/propietary/raaw/**/*b0*nc"
config_file = "C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/AWAKEN/LIDARGO_samples/config/config_raaw_b1.xlsx"
save_path='C:/Users/SLETIZIA/OneDrive - NREL/Desktop/241220_lidargo_test/raw'



#%% Initalizaipon
config_stand=pd.read_excel(config_file).set_index('regex')

#match standardized config

files=glob.glob(source)


#%% Main
for f in files:
    date_source=np.int64(re.search(r'\d{8}.\d{6}',f).group(0)[:8])
    for regex in config_stand.columns:
        match = re.findall(regex, f)
        sdate=config_stand[regex]['start_date']
        edate=config_stand[regex]['end_date']
        if len(match)==1 and date_source>=sdate and date_source<=edate:
            config = lg.LidarConfig(**config_stand[regex].to_dict())
        
            # Run processing
            lproc = lg.Standardize(f, config=config, verbose=True)
            lproc.process_scan(replace=False, save_file=True, make_figures=True,save_figures=True,save_path=save_path)
            
            plt.close('all')
