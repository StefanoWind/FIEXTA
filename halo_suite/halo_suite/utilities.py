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
                       kinematic_file: str=''):
    
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
        if kinematic_file=='':
            ppd1=config['ppd_azi'][lidar_id]#points per degree in azimuth motor
            ppd2=config['ppd_ele'][lidar_id]#points per degree in elevation motor
            
            dazi=((azi[1:] - azi[:-1] + 180) % 360) - 180
            dele=((ele[1:] - ele[:-1] + 180) % 360) - 180
            stop=np.concatenate(([0],np.where(np.abs((np.diff(dazi)+np.diff(dele)))>10**-10)[0]+1,[-1]))
            
            azi_range=azi[stop]
            P1=-azi_range*ppd1
            S1=S_azi*ppd1/10+np.zeros(len(P1))
            A1=A_azi+np.zeros(len(P1))
            a1=A1*1000/ppd1
            
            ele_range=ele[stop]
            P2=-ele_range*ppd2
            S2=S_ele*ppd2/10+np.zeros(len(P2))
            A2=A_ele+np.zeros(len(P2))
            a2=A2*1000/ppd2
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
            

        
                       