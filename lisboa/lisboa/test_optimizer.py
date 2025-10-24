# -*- coding: utf-8 -*-
"""
Test LiSBOA
"""
import numpy as np
from lisboa import scan_optimizer as opt
from matplotlib import pyplot as plt
plt.close('all')

#%% Inputs
azi1=[70,80]
azi2=[110,100]
ele1=[0,0]
ele2=[0,0]
dazi=[1,2,3]
dele=[0,0,0]
ppr=1000
dr=12
rmin=100
rmax=1000
T=600
tau=5
path_config_lidar='C:/Users/sletizia/Software/FIEXTA/halo_suite/halo_suite/configs/config.217.yaml'

config={'sigma':0.25,
        'mins':[0,-200],
        'maxs':[1000,200],
        'Dn0':[200,50],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1,
        'max_iter':3}

#%% Initialization

#%% Main
scopt=opt.scan_optimizer(config)
e1,e2=scopt.pareto(azi1, azi2, ele1, ele2, dazi, dele, volumetric=False,rmin=rmin,rmax=rmax, T=T,tau=tau,
                   mode='CSM', ppr=ppr, dr=dr, path_config_lidar=path_config_lidar)






