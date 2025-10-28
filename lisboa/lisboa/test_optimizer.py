# -*- coding: utf-8 -*-
"""
Test LiSBOA
"""
import numpy as np
from lisboa import scan_optimizer as opt
from matplotlib import pyplot as plt
plt.close('all')

#%% Inputs
coords='xy'
azi1=[70,80]
azi2=[110,100]
ele1=[-20,-10]
ele2=[20,10]
dazi=[1,2,4]
dele=[1,2,4]
ppr=1000
dr=12
rmin=100
rmax=1000
T=600
tau=5
path_config_lidar='C:/Users/sletizia/Software/FIEXTA/halo_suite/halo_suite/configs/config.217.yaml'

config={'sigma':0.25,
        'mins':[0,-200,-100],
        'maxs':[1000,200,200],
        'Dn0':[200,50,50],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1,
        'max_iter':3}

#%% Initialization


#%% Main
scopt=opt.scan_optimizer(config)
epsilon1,epsilon2,grid,Dd,excl,points,duration=scopt.pareto(coords,azi1, azi2, ele1, ele2, dazi, dele, volumetric=True,rmin=rmin,rmax=rmax, T=T,tau=tau,
                   mode='CSM', ppr=ppr, dr=dr, path_config_lidar=path_config_lidar)

#%% Plots
cmap = plt.cm.jet
colors = [cmap(v) for v in np.linspace(0,1,len(dazi))]
fig=plt.figure(figsize=(18,8))
N_row=int(np.floor(len(azi1)**0.5))
N_col=int(np.ceil(len(azi1)/N_row))
for i_ang in range(len(azi1)):
    ax=plt.subplot(N_row,N_col,i_ang+1)
    plt.plot(epsilon1[np.arange(len(azi1))!=i_ang,:],epsilon2[np.arange(len(azi1))!=i_ang,:],'.k',markersize=30,alpha=0.25)
    for i_dang in range(len(dazi)):
        plt.plot(epsilon1[i_ang,i_dang],epsilon2[i_ang,i_dang],'.',
                 color=colors[i_dang],markeredgecolor='k',markersize=30,
                 label=r'$\Delta \alpha='+str(dazi[i_dang])+r'^\circ$, $\Delta \beta='+str(dele[i_dang])+r'^\circ$')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel(r'$\epsilon_I$')
    plt.xlabel(r'$\epsilon_{II}$')  
    plt.grid()
plt.legend(draggable=True)


for i_ang in range(len(azi1)):
    fig=plt.figure(figsize=(18,8))
    N_row=int(np.floor(len(dazi)**0.5))
    N_col=int(np.ceil(len(dazi)/N_row))
    for i_dang in range(len(dazi)):
       
                          
        
        fill=np.zeros(np.shape(excl[i_ang,i_dang]))
        fill[excl[i_ang,i_dang]==False]=10
        
        if coords=='xy':
            ax=plt.subplot(N_row,N_col,i_dang+1)
            plt.pcolor(grid[i_ang,i_dang][0],grid[i_ang,i_dang][1],fill.T,cmap='Greys',vmin=0,vmax=1,alpha=0.5)
            plt.plot(points[i_ang,i_dang][0],points[i_ang,i_dang][1],'.',color=colors[i_dang],markersize=2)
            ax.set_aspect('equal')
            plt.xlim([config['mins'][0],config['maxs'][0]])
            plt.ylim([config['mins'][1],config['maxs'][1]])
        elif coords=='xyz':
            ax=plt.subplot(N_row,N_col,i_dang+1,projection='3d')
            ax.voxels(grid[i_ang,i_dang][0],grid[i_ang,i_dang][1],grid[i_ang,i_dang][2],color='k',alpha=0.5)
            plt.plot(points[i_ang,i_dang][0],points[i_ang,i_dang][1],'.',color=colors[i_dang],markersize=2)
            ax.set_aspect('equal')
            plt.xlim([config['mins'][0],config['maxs'][0]])
            plt.ylim([config['mins'][1],config['maxs'][1]])
        
            

    




