# -*- coding: utf-8 -*-
"""
Plot scan trajectory and compare to lidar simulation
"""

import os
cd=os.path.dirname(__file__)
from matplotlib import pyplot as plt
import yaml
from halo_suite.utilities import read_hpl
from halo_suite import halo_simulator as hls
import numpy as np
plt.close('all')

#%% Inputs
source=input('Data file: ')
scan_file=input('Scan file: ')
path_config=input('Config file: ')

#%% Initialization

with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)   

#%% Main

tnum,azi,ele,Nr,dr,ppr,mode=read_hpl(source,config)
t=tnum-tnum[0]

if mode=='ssm':
    #simulate scanning head
    halo_sim=hls.halo_simulator(config={'processing_time':config['Dt_p_SSM'],
                                         'acquisition_time':config['Dt_a_SSM'],
                                         'dwell_time': 0,
                                         'max_S_azi':config['S_azi_SSM'],
                                         'max_A_azi':config['A_azi_SSM'],
                                         'max_S_ele':config['S_ele_SSM'],
                                         'max_A_ele':config['A_ele_SSM']})
    
    t2,azi2,ele2,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode='ssm',ppr=ppr,source=scan_file,azi0=azi[0],ele0=ele[0])
    
elif mode=='csm':
    
    #simulate scanning head
    halo_sim=hls.halo_simulator(config={'processing_time':config['Dt_p_CSM'],
                                        'acquisition_time':config['Dt_a_CSM'],
                                        'dwell_time':config['Dt_d'][ppr],
                                        'ppd_azi':config['ppd_azi'],
                                        'ppd_ele':config['ppd_ele']})
    
    t2,azi2,ele2,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode='csm',ppr=ppr,source=scan_file,azi0=azi[0],ele0=ele[0])
    
plt.figure(figsize=(18,8))
ax=plt.subplot(2,1,1)
plt.plot(t_all,azi_all,'-r')
plt.plot(t,azi,'.k',label='Data')
plt.plot(t2,azi2,'o', markerfacecolor='none', markeredgecolor='r',label='Simulator')
plt.ylabel(r'$\alpha$ [$^\circ$]')
plt.title(f'PPR={ppr}, mode={mode}, file: {source}')
plt.legend()
plt.grid()

ax=plt.subplot(2,1,2)
plt.plot(t_all,ele_all,'-r')
plt.plot(t,ele,'.k')
plt.plot(t2,ele2,'o', markerfacecolor='none', markeredgecolor='r',label='Simulator')
plt.xlabel('Time [s]')
plt.ylabel(r'$\beta$ [$^\circ$]')
plt.grid()
    