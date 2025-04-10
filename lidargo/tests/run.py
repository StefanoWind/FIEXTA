# -*- coding: utf-8 -*-
"""
Test block
"""
import lidargo as lg
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

source = "../data/test/roof.lidar.z01.a0.20250319.045000.user5.nc"
config_file = "C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/G3P3/g3p3_lidar_processing/configs/config_g3p3_b0.xlsx"

#standardize
lproc = lg.Standardize(source, config=config_file, verbose=True)
lproc.process_scan(replace=True, save_file=False, make_figures=True)