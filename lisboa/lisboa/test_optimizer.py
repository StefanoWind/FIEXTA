# -*- coding: utf-8 -*-
"""
Test LiSBOA
"""
from lisboa import scan_optimizer as opt
from matplotlib import pyplot as plt

plt.close('all')

#%% Inputs
coords='xy'
azi1=[0,30]
azi2=[180,150]
ele1=[0,0]
ele2=[0,0]
dazi=[1,2,4,6,12]
dele=[0,0,0,0,0]
ppr=1000
dr=12
rmin=100
rmax=1000
T=600
tau=5
path_config_lidar='C:/Users/sletizia/Software/FIEXTA/halo_suite/halo_suite/configs/config.217.yaml'
volumetric=False
mode='CSM'

config={'sigma':0.25,
        'mins':[100,-1000],
        'maxs':[1000,1000],
        'Dn0':[100,25],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1,
        'max_iter':3}

#%% Initialization
scopt=opt.scan_optimizer(config)

#%% Main
Pareto=scopt.pareto(coords,azi1, azi2, ele1, ele2, dazi, dele, volumetric=volumetric,rmin=rmin,rmax=rmax, T=T,tau=tau,
                   mode=mode, ppr=ppr, dr=dr, path_config_lidar=path_config_lidar)