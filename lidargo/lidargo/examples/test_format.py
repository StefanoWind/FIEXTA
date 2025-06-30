'''
Test LIDARGO
'''
import os
cd = os.getcwd()
pwd = os.path.dirname(cd)

import lidargo as lg
from matplotlib import pyplot as plt
import warnings
import matplotlib

warnings.filterwarnings("ignore")
plt.close("all")

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.size"] = 14

# %% Inputs

# paths
filename = os.path.join(pwd, "data/test/User5_257_20250314_000012.hpl")

source_config_format= 'C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/G3P3/g3p3_lidar_processing/configs/config_g3p3_a0.xlsx'

# %% Main

#formatting
lproc = lg.Format(filename, config=source_config_format, verbose=True)
lproc.process_scan(replace=True, save_file=True)