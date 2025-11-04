# -*- coding: utf-8 -*-
"""
Test LiSBOA
"""
import numpy as np
from lisboa import scan_optimizer as opt
from matplotlib import pyplot as plt
plt.close('all')

#%% Inputs
coords='xyz'
azi1=[70,80]
azi2=[110,100]
ele1=[-20,-10]
ele2=[20,10]
dazi=[1,2,4]
dele=[1,2,4]
ppr=1000
dr=12
rmin=100
rmax=500
T=600
tau=5
path_config_lidar='C:/Users/sletizia/Software/FIEXTA/halo_suite/halo_suite/configs/config.217.yaml'
volumetric=True
mode='CSM'

config={'sigma':0.25,
        'mins':[100,-200,-100],
        'maxs':[500,200,200],
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
epsilon1,epsilon2,grid,Dd,excl,points,duration=scopt.pareto(coords,azi1, azi2, ele1, ele2, dazi, dele, volumetric=volumetric,rmin=rmin,rmax=rmax, T=T,tau=tau,
                   mode=mode, ppr=ppr, dr=dr, path_config_lidar=path_config_lidar)

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
    plt.ylabel(r'$\epsilon_{II}$')  
    plt.title(r'$\alpha \in ['+str(azi1[i_ang])+', '+str(azi2[i_ang])+r']^\circ$, $\beta \in ['+str(ele1[i_ang])+r', '+str(ele2[i_ang])+']^\circ$')
    plt.grid()
plt.legend(draggable=True)

#plot all scan geometries
for i_ang in range(len(azi1)):
    fig=plt.figure(figsize=(18,8))
    N_row=int(np.floor(len(dazi)**0.5))
    N_col=int(np.ceil(len(dazi)/N_row))
    for i_dang in range(len(dazi)):  
        info=r'$\alpha='+str(azi1[i_ang])+':'+str(dazi[i_dang])+':'+str(azi2[i_ang])+'^\circ$'+ '\n'+\
             r'$\beta=' +str(ele1[i_ang])+':'+str(dele[i_dang])+':'+str(ele2[i_ang])+'^\circ$' + '\n'+\
             r'$\Delta n_0='+str(config['Dn0'])+r'$ m'                                         + '\n'+\
             r'$\epsilon_I='+str(np.round(epsilon1[i_ang,i_dang],2))+r'$'                      + '\n'+\
             r'$\epsilon_{II}='+str(np.round(epsilon2[i_ang,i_dang],2))+r'$'                   + '\n'+\
             r'$\tau_s='+str(np.round(duration[i_ang,i_dang],1))+r'$ s'
    
        if coords!='xyz':
            fill=np.zeros(np.shape(excl[i_ang,i_dang]))
            fill[excl[i_ang,i_dang]==False]=10
            
            ax=plt.subplot(N_row,N_col,i_dang+1)
            plt.pcolor(grid[i_ang,i_dang][0],grid[i_ang,i_dang][1],fill.T,cmap='Greys',vmin=0,vmax=1,alpha=0.5)
            plt.plot(points[i_ang,i_dang][0],points[i_ang,i_dang][1],'.',color=colors[i_dang],markersize=2)
            ax.set_aspect('equal')
            ax.set_xlim([config['mins'][0],config['maxs'][0]])
            ax.set_ylim([config['mins'][1],config['maxs'][1]])
            ax.set_xlabel(r'$'+str(coords[0])+'$ [m]')
            ax.set_ylabel(r'$'+str(coords[1])+'$ [m]')
            dtick=np.max([np.diff(ax.get_xticks())[0],
                          np.diff(ax.get_yticks())[0]])
            ax.set_xticks(np.arange(config['mins'][0],config['maxs'][0]+dtick,dtick))
            ax.set_yticks(np.arange(config['mins'][1],config['maxs'][1]+dtick,dtick))
            ax.text(0,config['maxs'][1]/2,s=info,bbox={'edgecolor':'k','facecolor':(1,1,1,0.25)})

        elif coords=='xyz':
            fill=excl[i_ang,i_dang]==False
            ax=plt.subplot(N_row,N_col,i_dang+1,projection='3d')
            dx=np.diff(grid[i_ang,i_dang][0])[0]
            dy=np.diff(grid[i_ang,i_dang][1])[0]
            dz=np.diff(grid[i_ang,i_dang][2])[0]
            X,Y,Z=np.meshgrid(np.append(grid[i_ang,i_dang][0]-dx/2,grid[i_ang,i_dang][0][-1]+dx),
                      np.append(grid[i_ang,i_dang][1]-dy/2,grid[i_ang,i_dang][1][-1]+dy),
                      np.append(grid[i_ang,i_dang][2]-dz/2,grid[i_ang,i_dang][2][-1]+dz),indexing='ij')
            ax.voxels(X,Y,Z,fill,facecolor=colors[i_dang],alpha=0.1)
            sel_x=(points[i_ang,i_dang][0]>config['mins'][0])*(points[i_ang,i_dang][0]<config['maxs'][0])
            sel_y=(points[i_ang,i_dang][1]>config['mins'][1])*(points[i_ang,i_dang][1]<config['maxs'][1])
            sel_z=(points[i_ang,i_dang][2]>config['mins'][2])*(points[i_ang,i_dang][2]<config['maxs'][2])
            sel=sel_x*sel_y*sel_z
            plt.plot(points[i_ang,i_dang][0][sel],points[i_ang,i_dang][1][sel],points[i_ang,i_dang][2][sel],'.k',markersize=2)
            ax.set_aspect('equal')
            ax.set_xlim([config['mins'][0],config['maxs'][0]])
            ax.set_ylim([config['mins'][1],config['maxs'][1]])
            ax.set_zlim([config['mins'][2],config['maxs'][2]])
            ax.set_xlabel(r'$x$ [m]')
            ax.set_ylabel(r'$y$ [m]')
            ax.set_zlabel(r'$z$ [m]')
            dtick=np.max([np.diff(ax.get_xticks())[0],
                          np.diff(ax.get_yticks())[0],
                          np.diff(ax.get_zticks())[0]])
            ax.set_xticks(np.arange(config['mins'][0],config['maxs'][0]+dtick,dtick))
            ax.set_yticks(np.arange(config['mins'][1],config['maxs'][1]+dtick,dtick))
            ax.set_zticks(np.arange(config['mins'][2],config['maxs'][2]+dtick,dtick))
            ax.text(0,0,config['maxs'][2]/2,s=info,bbox={'edgecolor':'k','facecolor':(1,1,1,0.25)})
