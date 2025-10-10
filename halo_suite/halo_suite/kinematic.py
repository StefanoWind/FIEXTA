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
from halo_suite import halo_simulator as hls
import glob
plt.close('all')

#%% Inputs
path_config=os.path.join(cd,'configs/config.{lidar_id}.yaml')
lidar_id=input('Lidar ID: ')
mode=input('Mode (SSM or CSM): ').lower()

#%% Initialization
#configs
with open(path_config.format(lidar_id=lidar_id), 'r') as fid:
    config = yaml.safe_load(fid)      

def motion_time(dang,S,A):
    '''
    Motion time as a function of the angular step, speed, and acceleration
    '''
    dt=dang+np.nan
    ang_crit=np.abs(S**2/A)
    dt[np.abs(dang)<ang_crit]=2*(np.abs(dang)[np.abs(dang)<ang_crit]/A)**0.5
    dt[np.abs(dang)>=ang_crit]=np.abs(dang)[np.abs(np.abs(dang))>=ang_crit]/S+S/A
    
    return dt
    
#%% Main

#STATIC ACQUISITION
files=glob.glob(os.path.join(config['path_data'],f'{lidar_id}','kinematic','acquisition',f'{mode}','*hpl'))
if len(files)==0:
    #write file for acquisitin test
    if mode=='ssm':
        scan_file_compiler(mode=mode,azi=[0],ele=[0],repeats=config['repeats_test'],
                           identifier=f'acquisition.{lidar_id}',config=config)
    elif mode=='csm':
        scan_file_compiler(mode=mode,azi=[0,360],ele=[0,0],repeats=1,
                           identifier=f'acquisition.{lidar_id}',config=config)
    print(f'File for acquisition test saved as ./scans/acquisition.{lidar_id}.txt. Run it on the lidar with different PPRs. Save data in ./data/{lidar_id}/kinematic/acquisition/{mode}')
else:
    #read acquisition test data
    dDt_avg=[]
    dt_low=[]
    dt_top=[]
    ppr_all=[]
    for f in files:
        tnum,azi,ele,Nr,dr,ppr=read_hpl(f,config)
        dDt_avg=np.append(dDt_avg,np.nanmedian(np.diff(tnum[1:-1])))
        dt_low=np.append(dt_low,np.nanpercentile(np.diff(tnum[1:-1]),5))
        dt_top=np.append(dt_top,np.nanpercentile(np.diff(tnum[1:-1]),95))
        ppr_all=np.append(ppr_all,ppr)
        
    LF=np.polyfit(ppr_all,dDt_avg,1)
    Dt_p=np.round(LF[1],4)
    Dt_a=np.round(LF[0]*1000,4)/1000
    
    #plot linear fit
    plt.figure()
    plt.plot(ppr_all,dDt_avg,'.k',markersize=10,label='Data')
    plt.errorbar(ppr_all,dDt_avg,[dDt_avg-dt_low,dt_top-dDt_avg],fmt='o',color='k',capsize=5)
    plt.plot(ppr_all,Dt_p+Dt_a*ppr_all,'-r',label=r'$'+str(Dt_p)+'+'+str(Dt_a*1000)+r'\cdot 10^{-3}$ PPR',zorder=10)
    plt.xlabel('PPR')
    plt.ylabel(r'$\Delta t$ [s]')
    plt.title(f'{mode}')
    plt.grid()
    plt.legend()
    
    #DYNAMIC ACQUISITION
    
    #prepare scan sequence
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
    
    if mode=='ssm':
        scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=1,identifier=f'motion.{lidar_id}')
        print(f'File for motion test saved as ./scans/motion.{lidar_id}.txt. Run it on the lidar with different PPRs. Save data in ./data/{lidar_id}/kinematic/motion/{mode}')
    elif mode=='csm':
        ppr_test=int(input('PPR for motion test: '))
        
        #prepare configuration
        config_csm={}
        for c in ['ppd_azi','ppd_ele','S_max_azi','S_max_ele','A_max_azi','A_max_ele','ang_tol']:
            config_csm[c]=config[c]
        config_csm['Dt_p']=Dt_p
        config_csm['Dt_a']=Dt_a
        config_csm['Dt_d']=0
        scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=1,identifier=f'motion.{lidar_id}.{ppr_test}',ppr=ppr_test,
                           config=config_csm)
        print(f'File for motion test saved as ./scans/motion.{lidar_id}.{ppr_test}.txt. Run it on the lidar with the selected PPRs. Save data in ./data/{lidar_id}/kinematic/motion/{mode}')

    #read motion test data
    files=glob.glob(os.path.join(config['path_data'],f'{lidar_id}','kinematic','motion',f'{mode}','*hpl'))
    for f in files:
        tnum,azi,ele,Nr,dr,ppr=read_hpl(f,config)
        if mode=='csm':
            if ppr!=ppr_test:
                continue
        azi=np.round(azi/config['ang_tol'])*config['ang_tol']%360
        ele=np.round(ele/config['ang_tol'])*config['ang_tol']%360
        t=tnum-tnum[0]
        Dt_m=np.diff(t)
        dazi=((azi[1:] - azi[:-1] + 180) % 360) - 180
        dele=((ele[1:] - ele[:-1] + 180) % 360) - 180
        
        if mode=='ssm':
            #fit kinematic model to estimate kinematic parameters of SSM
            V, cov = curve_fit(motion_time,dazi[dazi>0.01],Dt_m[dazi>0.01]-Dt_p-Dt_a*ppr)
            S_azi=np.round(V[0],1)
            A_azi=np.round(V[1],1)
            
            V, cov = curve_fit(motion_time,dele[dele>0.01],Dt_m[dele>0.01]-Dt_p-Dt_a*ppr)
            S_ele=np.round(V[0],1)
            A_ele=np.round(V[1],1)
            
            #plot
            plt.figure(figsize=(16,8))
            dazi_plot=np.arange(0,np.nanmax(dazi),0.1)
            dele_plot=np.arange(0,np.nanmax(dele),0.1)
            ax=plt.subplot(2,1,1)
            plt.plot(dazi[dazi>0.01],Dt_m[dazi>0.01]-Dt_p-Dt_a*ppr,'.k',alpha=0.5,label='PPI data')
            plt.plot(dazi_plot,motion_time(dazi_plot,S_azi, A_azi),'-r',
                     label=r'$\dot{\alpha}_{max}='+str(S_azi)+r'^\circ s^{-1}$'+'\n'+r'$\ddot{\alpha}='+str(A_azi)+r'^\circ s^{-2}$')
            plt.xlabel(r'$\Delta \alpha$ [$^\circ$]')
            plt.ylabel(r'$\Delta t_m$ [s]')
            plt.legend()
            plt.grid()
            
            plt.title(f'PPR={ppr}, mode={mode}, file: {os.path.basename(f)}')
            ax=plt.subplot(2,1,2)
            plt.plot(dele[dele>0.01],Dt_m[dele>0.01]-Dt_p-Dt_a*ppr,'.k',alpha=0.5,label='RHI data')
            plt.plot(dele_plot,motion_time(dele_plot,S_ele, A_ele),'-r',
                     label=r'$\dot{\beta}_{max}='+str(S_ele)+r'^\circ s^{-1}$'+'\n'+r'$\ddot{\beta}='+str(A_ele)+r'^\circ s^{-2}$')
            plt.xlabel(r'$\Delta \beta$ [$^\circ$]')
            plt.ylabel(r'$\Delta t_m$ [s]')
            plt.legend()
            plt.grid()

            #simulate scanning head
            halo_sim=hls.halo_simulator(config={'processing_time':Dt_p,
                                                 'acquisition_time':Dt_a,
                                                 'dwell_time': 0,
                                                 'max_S_azi':S_azi,
                                                 'max_A_azi':A_azi,
                                                 'max_S_ele':S_ele,
                                                 'max_A_ele':A_ele})
            
            t2,azi2,ele2,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode='ssm',ppr=ppr,azi=azi,ele=ele)
            
        elif mode=='csm':
            
            #simulate scanning head
            halo_sim=hls.halo_simulator(config={'processing_time':Dt_p,
                                                'acquisition_time':Dt_a,
                                                'dwell_time':config['Dt_d'][ppr],
                                                'ppd_azi':config['ppd_azi'],
                                                'ppd_ele':config['ppd_ele']})
            
            t2,azi2,ele2,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode='csm',ppr=ppr,
                                                                       source=os.path.join(cd,'scans',f'motion.{lidar_id}.{ppr}.csm.txt'))
            
            plt.figure(figsize=(18,8))            
            ax=plt.subplot(2,1,1)
            for da in config['dazi_test']:
                if da==config['dazi_test'][0]:
                    plt.plot(t2,t2*0+da,'--r',label='Targets')
                else:
                    plt.plot(t2,t2*0+da,'--r')
            plt.plot(t[1:],dazi,'.k',label='Data')
            ax.set_yscale('symlog')
            plt.ylim([0,np.max(config['dazi_test'])+5])
            plt.ylabel(r'$\Delta\alpha$ [$^\circ$]')
            plt.title(f'PPR={ppr}, mode={mode}, file: {os.path.basename(f)}')
            plt.legend()
            plt.grid()
            
            ax=plt.subplot(2,1,2)
            for da in config['dele_test']:
                plt.plot(t2,t2*0+da,'--r')
            plt.plot(t[1:],dele,'.k',label='Data')
            ax.set_yscale('symlog')
            plt.ylim([0,np.max(config['dele_test'])+5])
            plt.ylabel(r'$\Delta\beta$ [$^\circ$]')
            plt.title(f'PPR={ppr}, mode={mode}, file: {os.path.basename(f)}')
            plt.xlabel('Time [s]')
            plt.grid()
            
        plt.figure(figsize=(18,8))
        ax=plt.subplot(2,1,1)
        plt.plot(t_all,azi_all,'-r')
        plt.plot(t,azi,'.k',label='Data')
        plt.plot(t2,azi2,'o', markerfacecolor='none', markeredgecolor='r',label='Simulator')
        plt.ylabel(r'$\alpha$ [$^\circ$]')
        plt.title(f'PPR={ppr}, mode={mode}, file: {os.path.basename(f)}')
        plt.legend()
        plt.grid()
        
        ax=plt.subplot(2,1,2)
        plt.plot(t_all,ele_all,'-r')
        plt.plot(t,ele,'.k')
        plt.plot(t2,ele2,'o', markerfacecolor='none', markeredgecolor='r',label='Simulator')
        plt.xlabel('Time [s]')
        plt.ylabel(r'$\beta$ [$^\circ$]')
        plt.grid()
        
add=input('Add parameters to configuration? (y/n): ')
if add.lower()=='y':
    if mode=='ssm':
        config['Dt_p_SSM']=Dt_p.item()
        config['Dt_a_SSM']=Dt_a.item()
        config['S_azi_SSM']=S_azi.item()
        config['S_ele_SSM']=S_ele.item()
        config['A_azi_SSM']=A_azi.item()
        config['A_ele_SSM']=A_ele.item()
    elif mode=='csm':
        config['Dt_p_CSM']=Dt_p.item()
        config['Dt_a_CSM']=Dt_a.item()

with open(path_config.format(lidar_id=lidar_id), 'w') as fid:
    yaml.safe_dump(config,fid)      