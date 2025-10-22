# -*- coding: utf-8 -*-
"""
Lidar Statistical Barnes Objective Analysis
"""

import numpy as np
from halo_suite import halo_simulator as hls 
from halo_suite.utilities import scan_file_compiler
from lisboa import statistics as stats
import yaml

class scan_optimizer:
    def __init__(self,
                 config: dict):

        self.config=config
      
    def pareto(
            self,
            azi1: np.array,
            azi2: np.array,
            ele1: np.array,
            ele2: np.array,
            dazi: np.array,
            dele: np.array,
            volumetric: bool,
            mode: str,
            ppr: int,
            dr: float,
            path_config_lidar: str):
        
        #zeroing
        epsilon1=np.array(azi1)+np.nan
        epsilon2=np.array(azi1)+np.nan
        
        with open(path_config_lidar, 'r') as fid:
            config_lidar = yaml.safe_load(fid)    
        
        
        for a1,a2,e1,e2 in zip(azi1,azi2,ele1,ele2):
            for da,de in zip(dazi,dele):
                da+=10**-10
                de+=10**-10
                azi=np.arange(a1,a2+da/2,da)
                ele=np.arange(e1,e2+de/2,de)
                if len(azi)==1:
                    azi=azi.squeeze()+np.zeros(len(ele))
                if len(ele)==1:
                    ele=ele.squeeze()+np.zeros(len(azi))
                
                azi=np.append(azi,azi[0])
                ele=np.append(ele,ele[0])
                if mode=='csm':
                    scan_file=scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=1,ppr=ppr,volumetric=volumetric,
                                       identifier=f'{a1}.{a2}.{e1}.{e2}.{ppr}',config=config_lidar)
       
                    halo_sim=hls.halo_simulator(config={'processing_time':config_lidar['Dt_d'][ppr],
                                                        'acquisition_time':config_lidar['Dt_a_CSM'],
                                                        'dwell_time':config_lidar['Dt_d'][ppr],
                                                        'ppd_azi':config_lidar['ppd_azi'],
                                                        'ppd_ele':config_lidar['ppd_ele']})
                
                    t2,azi2,ele2,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode='csm',ppr=ppr,
                                                                               source=scan_file,azi0=azi[0],ele0=ele[0])
                        
                    
        return epsilon1,epsilon2