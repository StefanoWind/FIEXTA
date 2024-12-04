"""
Run all examples
"""

import os
import pandas as pd
import sys
sys.path.append('../../lidargo')

import LIDARGO_standardize as stand
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
source = os.path.join("../../data/lidargo/**/*nc")
source_config_stand = os.path.join('../../configs/lidargo/config_examples_stand.xlsx')
source_config_stats = os.path.join('../../configs/config_examples_stats.xlsx')

#%% Initialization
files=glob.glob(source,recursive=True)

#%% Main
for f in files:
    lproc = stand.LIDARGO(f,source_config_stand, verbose=True)
    lproc.process_scan(replace=False,save_file=True,make_figures=False)
            
      