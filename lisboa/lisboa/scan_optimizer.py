# -*- coding: utf-8 -*-
"""
Lidar Statistical Barnes Objective Analysis
"""

import numpy as np
from halo_suite import halo_simulator as hls 
from halo_suite.utilities import scan_file_compiler
from lisboa import statistics as stats
import lisboa.utilities as utl
from matplotlib import pyplot as plt
import yaml

class scan_optimizer:
    def __init__(self,
                 config: dict):

        self.config=config
      
    def pareto(
            self,
            coords,
            azi1: np.array,
            azi2: np.array,
            ele1: np.array,
            ele2: np.array,
            dazi: np.array,
            dele: np.array,
            rmin: float,
            rmax: float,
            dr: float,
            ppr: int,
            volumetric: bool,
            mode: str,
            path_config_lidar: str,
            T: float,
            tau: float):
        
        #zeroing
        epsilon1=np.zeros((len(azi1),len(dazi)))+np.nan
        epsilon2=np.zeros((len(azi1),len(dazi)))+np.nan
        grid=np.empty((len(azi1),len(dazi)),dtype=object)
        Dd=np.empty((len(azi1),len(dazi)),dtype=object)
        excl=np.empty((len(azi1),len(dazi)),dtype=object)
        points=np.empty((len(azi1),len(dazi)),dtype=object)
        duration=np.zeros((len(azi1),len(dazi)))+np.nan
        
        #initialize LiSBOA
        lproc=stats.statistics(self.config)
        
        with open(path_config_lidar, 'r') as fid:
            config_lidar = yaml.safe_load(fid)    
        
        r=np.arange(rmin,rmax+dr,dr)
        i_ang=0
        for a1,a2,e1,e2 in zip(azi1,azi2,ele1,ele2):
            i_dang=0
            for da,de in zip(dazi,dele):
                print(f"Evaluating azi={a1}:{da}:{a2},ele={e1}:{de}:{e2}")
                da+=10**-10
                de+=10**-10
                azi=np.arange(a1,a2+da/2,da)
                ele=np.arange(e1,e2+de/2,de)
                if len(azi)==1:
                    azi=azi.squeeze()+np.zeros(len(ele))
                if len(ele)==1:
                    ele=ele.squeeze()+np.zeros(len(azi))
                
                if mode=='CSM':
                    scan_file=scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=1,ppr=ppr,
                                       identifier=f'{a1:.2f}.{da:.2f}.{a2:.2f}.{e1:.2f}.{de:.2f}.{e2:.2f}.{ppr}',config=config_lidar,optimize=True,volumetric=True,reset=True)
       
                    halo_sim=hls.halo_simulator(config={'processing_time':config_lidar['Dt_p_CSM'],
                                                        'acquisition_time':config_lidar['Dt_a_CSM'],
                                                        'dwell_time':config_lidar['Dt_d_CSM'][ppr],
                                                        'ppd_azi':config_lidar['ppd_azi'],
                                                        'ppd_ele':config_lidar['ppd_ele']})
                
                    t,azi_sim,ele_sim,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode=mode,ppr=ppr,source=scan_file)
                # elif mode=='SSM':
                    
                #epsilon1
                x,y,z=utl.sphere2cart(r, azi_sim, ele_sim)
                if coords=='xy':
                    x_exp=[x.ravel(),y.ravel()]
                elif coords=='xz':
                     x_exp=[x.ravel(),z.ravel()]
                elif coords=='yz':
                    x_exp=[y.ravel(),z.ravel()]
                elif coords=='xyz':
                    x_exp=[x.ravel(),y.ravel(),z.ravel()]
                grid[i_ang,i_dang],Dd[i_ang,i_dang],excl[i_ang,i_dang],_,_,_,_=lproc.calculate_weights(x_exp)
                epsilon1[i_ang,i_dang]=np.sum(excl[i_ang,i_dang])/np.size(excl[i_ang,i_dang])
                points[i_ang,i_dang]=x_exp
                
                #epsilon2
                L=np.floor(T/t[-1])
                p=np.arange(1,L)
                epsilon2[i_ang,i_dang]=(1/L+2/L**2*np.sum((L-p)*np.exp(-T/tau*p)))**0.5
                duration[i_ang,i_dang]=t[-1]
                i_dang+=1
            i_ang+=1
                    
        return epsilon1,epsilon2,grid,Dd,excl,points,duration