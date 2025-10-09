# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:29:35 2025

@author: sletizia
"""
import os
cd=os.path.dirname(__file__)
import os
import numpy as np
from datetime import datetime
from halo_suite import halo_simulator as hls
from scipy.optimize import minimize
def scan_file_compiler(mode: str, 
                       azi: np.array,
                       ele: np.array,
                       repeats: int,
                       save_path: str='',
                       identifier: str='',
                       volumetric: bool=False,
                       lidar_id: str='',
                       ppr: int=0,
                       S_azi: float=36,
                       S_ele: float=72,
                       A_azi: float=50,
                       A_ele: float=50,
                       config: dict={},
                       ang_tol: float=0.1):
    
    if save_path=='':
        save_path=os.path.join(cd,'scans')
    if identifier=='':
        identifier=datetime.strftime(datetime.now(),'%Y%m%d.%H%M%S')
    
    if volumetric:
        [azi,ele]=np.meshgrid(azi,ele)
        azi=azi.ravel()
        ele=ele.ravel()
        vol_flag='vol.'
    else:
        vol_flag=''
        
    if mode=='ssm':
        if isinstance(azi, np.ndarray):
            L=''
            for a,e in zip(azi[:-1],ele[:-1]):
                L=L+('%07.3f' % a+ '%07.3f' % e +'\n')
        else:
            L='%07.3f' % azi+ '%07.3f' % ele +'\n'
    
    if mode =='csm':
        if type(azi).__name__!= 'ndarray':
            if type(azi).__name__!='list':
                azi=[azi]
            azi=np.array(azi)
        if type(ele).__name__!= 'ndarray':
            if type(ele).__name__!='list':
                ele=[ele]
            ele=np.array(ele)
        # azi=np.append(azi,azi[0])
        # ele=np.append(ele,ele[0])
       
            
        ppd1=config['ppd_azi']#points per degree in azimuth motor
        ppd2=config['ppd_ele']#points per degree in elevation motor
        
        dazi=(azi[1:] - azi[:-1] + 180) % 360 - 180
        dele=(ele[1:] - ele[:-1] + 180) % 360 - 180
        stop=np.concatenate(([0],np.where(np.abs((np.diff(dazi)+np.diff(dele)))>10**-10)[0]+1,[-1]))
        
        azi_range=azi[stop]
        P1=-azi_range*ppd1
        ele_range=ele[stop]
        P2=-ele_range*ppd2
        
        if 'T_d' in config:
            T_d=config['T_d']
            T_a=config['T_dppr']
            S_max_azi=config['S_max_azi']*10/ppd1
            A_max_azi=config['A_max_azi']*1000/ppd1
            S_max_ele=config['S_max_ele']*10/ppd2
            A_max_ele=config['A_max_ele']*1000/ppd2
            T_s=T_d+T_a*ppr
            
            S_azi=[S_max_azi]
            A_azi=[A_max_azi]
            S_ele=[S_max_ele]
            A_ele=[A_max_ele]
            for i1,i2 in zip(stop[:-1:2],stop[1::2]):
                dazi=(azi[i1+1]-azi[i1]+ 180) % 360 - 180
                dele=(ele[i1+1]-ele[i1]+ 180) % 360 - 180
                if np.abs(dazi)>ang_tol and np.abs(dele)<ang_tol:
                    if dazi<=S_max_azi*T_s:
                        res = minimize(angular_error,[dazi/T_s,A_max_azi],
                                       args=(azi[i2]-azi[i1], dazi, T_d,T_a,ppr),
                                       bounds= [(ang_tol, S_max_azi), (ang_tol, A_max_azi)])
                        if res.success==False:
                            raise BaseException(f'Optimization of motion failed for azimuth step={dazi} deg, PPR={ppr}.')
                        opt=res.x
                    else:
                        opt=[S_max_azi,A_max_azi]
                    
                    S_azi=np.append(S_azi,opt[0])
                    S_ele=np.append(S_ele,0)
                    A_azi=np.append(A_azi,opt[1])
                    A_ele=np.append(A_ele,0)
                    
                    S_azi=np.append(S_azi,S_max_azi)
                    S_ele=np.append(S_ele,0)
                    A_azi=np.append(A_azi,A_max_azi)
                    A_ele=np.append(A_ele,0)
                   
                elif np.abs(dazi)<ang_tol and np.abs(dele)>ang_tol:
                    if dele<=S_max_ele*T_s:
                        res = minimize(angular_error,[dele/T_s,A_max_ele],
                                       args=(ele[i2]-ele[i1], dele, T_d,T_a,ppr),
                                       bounds= [(ang_tol, S_max_ele), (ang_tol, A_max_ele)])
                        if res.success==False:
                            raise BaseException(f'Optimization of motion failed for elevation step={dele} deg, PPR={ppr}.')
                        opt=res.x
                    else:
                        opt=[S_max_ele,A_max_ele]
                    
                    S_azi=np.append(S_azi,0)
                    S_ele=np.append(S_ele,opt[0])
                    A_azi=np.append(A_azi,0)
                    A_ele=np.append(A_ele,opt[1])
                    S_azi=np.append(S_azi,0)
                    S_ele=np.append(S_ele,S_max_ele)
                    A_azi=np.append(A_azi,0)
                    A_ele=np.append(A_ele,A_max_ele)
            S_azi=S_azi[:-1]
            S_ele=S_ele[:-1]
            A_azi=A_azi[:-1]
            A_ele=A_ele[:-1]
           
        S1=S_azi*ppd1/10+np.zeros(len(P1))
        A1=A_azi*ppd1/1000+np.zeros(len(P1))
        S2=S_ele*ppd2/10+np.zeros(len(P2))
        A2=A_ele*ppd2/1000+np.zeros(len(P2))
   
        L=''
        for p1,p2,s1,s2,a1,a2 in zip(P1,P2,S1,S2,A1,A2):
            
            l='A.1=%.0f'%a1+',S.1=%.0f'%s1+',P.1=%.0f'%p1+'*A.2=%.0f'%a2+',S.2=%.0f'%s2+',P.2=%.0f'%p2+'\nW=0\n'
            L+=l
        
    os.makedirs(save_path,exist_ok=True)
    save_name=f'{identifier}.{mode}.{vol_flag}txt'
    with open(os.path.join(save_path,save_name),'w') as fid:
        fid.write(L*repeats)
        fid.close()
        
        
def read_hpl(file,lidar_id,config):
    '''
    Read kinematic from hlp lidar file
    '''
    
    #read ranges
    with open(file, "r") as fid:
        Nr=int(fid.readlines()[config['hpl_Nr']['217']].split(':')[1].strip())
        fid.seek(0)
        dr=np.float64(fid.readlines()[config['hpl_dr']['217']].split(':')[1].strip())
        fid.seek(0)
        ppr=int(fid.readlines()[config['hpl_ppr']['217']].split(':')[1].strip())
        fid.seek(0)
        data=fid.readlines()[config['hpl_header'][lidar_id]::Nr+1]

        #zeroing (specific file)
        tnum=[]
        azi=[]
        ele=[]
         
        #read only first gate info (time, angles)
        for d in np.array(data):
            d_split=np.array(d.split(' '))
            d_split=d_split[d_split!='']
            tnum=np.append(tnum,np.float64(d_split[config['hpl_index_time'][lidar_id]])*3600)
            azi=np.append(azi,np.float64(d_split[config['hpl_index_azi'][lidar_id]]))
            ele=np.append(ele,np.float64(d_split[config['hpl_index_ele'][lidar_id]]))
                
        return tnum, azi, ele, Nr, dr, ppr
            

def angular_error(params,ang_range,dang,T_d,T_a,ppr,ang_tol=0.1):
    
    S=params[0]
    A=params[1]
    halo_sim=hls.halo_simulator(config={'processing_time':T_d,
                  'acquisition_time':T_a,
                  'S_ele':0,
                  'A_ele':0})
    
    T,ang,_,_,_,_=halo_sim.scanning_head_sim(mode='csm',ppr=ppr,azi=np.array([0,ang_range]),ele=np.array([0,0]),
                                                   S_azi=S,A_azi=A,S_ele=0,A_ele=0)
    dang2=(ang[1:]-ang[:-1]+ 180) % 360 - 180
    dang2=np.append(dang2[0],dang2)
    ang_error=(np.median(np.diff(ang[np.abs(dang2)>ang_tol]))-dang)**2
    return ang_error
    
        
                       