# -*- coding: utf-8 -*-
"""
Characterize kinematic of scanning head
"""
import os
cd=os.path.dirname(__file__)
from matplotlib import pyplot as plt
import numpy as np
import yaml
from halo_suite.utilities import scan_file_compiler, read_hpl
from scipy.optimize import curve_fit
import glob
plt.close('all')

#%% Inputs
path_config=os.path.join(cd,'configs/config.yaml')
lidar_id=input('Lidar ID: ')
mode=input('Mode (SSM or CSM): ')

#%% Initialization
#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)      

def kinematic_model(dang,S,A):
    dt=dang+np.nan
    ang_crit=S**2/A
    dt[dang<ang_crit]=2*(dang[dang<ang_crit]/A)**0.5
    dt[dang>=ang_crit]=dang[dang>=ang_crit]/S+S/A
    
    return dt
    
#%% Main

#static acquisition
files=glob.glob(os.path.join(config['path_data'],f'{lidar_id}','stares.kinematic',f'{mode}','*hpl'))
if len(files)==0:
    scan_file_compiler(mode=mode,azi=90,ele=90,repeats=config['repeats_test'],
                       identifier=f'Stare_{config["repeats_test"]}_points',config=config,lidar_id=lidar_id)
    print(f'Test stare file saved. Run it on the lidar with different PPRs. Save data in ./data/{lidar_id}/stares_kinematic')
else:
    dt_avg=[]
    dt_low=[]
    dt_top=[]
    ppr_all=[]
    for f in files:
        tnum,azi,ele,Nr,dr,ppr=read_hpl(f,lidar_id,config)
        dt_avg=np.append(dt_avg,np.nanmedian(np.diff(tnum)))
        dt_low=np.append(dt_low,np.nanpercentile(np.diff(tnum),5))
        dt_top=np.append(dt_top,np.nanpercentile(np.diff(tnum),95))
        ppr_all=np.append(ppr_all,ppr)
        
    LF=np.polyfit(ppr_all,dt_avg,1)
    dt_d=np.round(LF[1],4)
    dt_dppr=np.round(LF[0]*1000,4)/1000
    
    output={'dt_d':dt_d,
            'dt_dppr':dt_dppr}
    
    #plot linear fit
    plt.figure()
    plt.plot(ppr_all,dt_avg,'.k',markersize=10,label='Stare data')
    plt.errorbar(ppr_all,dt_avg,[dt_avg-dt_low,dt_top-dt_avg],fmt='o',color='k',capsize=5)
    plt.plot(ppr_all,dt_d+dt_dppr*ppr_all,'-r',label=r'$'+str(dt_d)+'+'+str(dt_dppr*1000)+r'\cdot 10^{-3}$ PPR',zorder=10)
    plt.xlabel('PPR')
    plt.ylabel(r'$\Delta t$ [s]')
    plt.title(f'{mode}')
    plt.grid()
    plt.legend()
    
#dynamic acquisition
files=glob.glob(os.path.join(config['path_data'],f'{lidar_id}','ppi_rhi.kinematic','*hpl'))
if len(files)==0:
    azi=[]
    ele=[]
    for dazi in config['dazi_test']:
        max_azi=config['azi_max_test']
        if dazi>=max_azi/2:
            max_azi=360
        new_azi=np.arange(0,max_azi,dazi+0.1)
        azi=np.append(azi,new_azi)
        ele=np.append(ele,new_azi*0)
        
    for dele in config['dele_test']:
        max_ele=config['ele_max_test']
        if dele>=max_ele/2:
            max_ele=180
        new_ele=np.arange(0,max_ele,dele+0.1)
        azi=np.append(azi,new_ele*0)
        ele=np.append(ele,new_ele)
        
    scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=1,identifier=f'PPI_RHI_{len(azi)}_points')
    print(f'Test PPI+RHI file saved. Run it on the lidar with different PPRs. Save data in ./data/{lidar_id}/ppi_rhi.kinematic')
else:
    for f in files:
        tnum,azi,ele,Nr,dr,ppr=read_hpl(f,lidar_id,config)
        dt_m=np.diff(tnum)
        dazi=((azi[1:] - azi[:-1] + 180) % 360) - 180
        dele=((ele[1:] - ele[:-1] + 180) % 360) - 180
        
        #fit kinematic model
        V, cov = curve_fit(kinematic_model,dazi[dazi>0.01],dt_m[dazi>0.01]-dt_d-dt_dppr*ppr)
        S_azi=np.round(V[0],1)
        A_azi=np.round(V[1],1)
        
        V, cov = curve_fit(kinematic_model,dele[dele>0.01],dt_m[dele>0.01]-dt_d-dt_dppr*ppr)
        S_ele=np.round(V[0],1)
        A_ele=np.round(V[1],1)
        
        plt.figure(figsize=(16,8))
        dazi_plot=np.arange(0,np.nanmax(dazi),0.1)
        dele_plot=np.arange(0,np.nanmax(dele),0.1)
        ax=plt.subplot(2,1,1)
        plt.plot(dazi[dazi>0.01],dt_m[dazi>0.01]-dt_d-dt_dppr*ppr,'.k',alpha=0.5,label='PPI data')
        plt.plot(dazi_plot,kinematic_model(dazi_plot,S_azi, A_azi),'-r',
                 label=r'$\dot{\alpha}_{max}='+str(S_azi)+r'^\circ s^{-1}$'+'\n'+r'$\ddot{\alpha}='+str(A_azi)+r'^\circ s^{-2}$')
        plt.xlabel(r'$\Delta \alpha$ [$^\circ$]')
        plt.ylabel(r'$\Delta t_m$ [s]')
        plt.legend()
        plt.grid()
        
        plt.title(f'PPR={ppr}, mode={mode}, file: {os.path.basename(f)}')
        ax=plt.subplot(2,1,2)
        plt.plot(dele[dele>0.01],dt_m[dele>0.01]-dt_d-dt_dppr*ppr,'.k',alpha=0.5,label='RHI data')
        plt.plot(dele_plot,kinematic_model(dele_plot,S_ele, A_ele),'-r',
                 label=r'$\dot{\beta}_{max}='+str(S_ele)+r'^\circ s^{-1}$'+'\n'+r'$\ddot{\beta}='+str(A_ele)+r'^\circ s^{-2}$')
        plt.xlabel(r'$\Delta \beta$ [$^\circ$]')
        plt.ylabel(r'$\Delta t_m$ [s]')
        plt.legend()
        plt.grid()
    

    

