# -*- coding: utf-8 -*-
"""
Standardize a specicic file
"""
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Software/FIEXTA/lidargo')
import config as cfg
import standardize as sd
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

source = "../../data/lidargo/example1/sc1.lidar.z01.a0.20230830.064613.user4.nc"
config_file = "../../configs/lidargo/config_examples_stand.xlsx"

config_stand=pd.read_excel(config_file).set_index('regex')

#match standardized config
date_source=np.int64(re.search(r'\d{8}.\d{6}',source).group(0)[:8])

matches=[]
for regex in config_stand.columns:
    match = re.findall(regex, source)
    sdate=config_stand[regex]['start_date']
    edate=config_stand[regex]['end_date']
    if len(match)>0 and date_source>=sdate and date_source<=edate:
        matches.append(regex)

if len(matches)==1:
    config =cfg.LidarConfig(**config_stand[matches[0]].to_dict())

    # Run processing
    lproc = sd.Standardize(source, config=config, verbose=True)
    lproc.process_scan(replace=True, save_file=True, make_figures=True)
