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
                       model: str='',
                       ppr: int=0):
    
    if save_path=='':
        save_path=os.path.join(cd,'scans')
    if identifier=='':
        identifier=datetime.strftime(datetime.now(),'%Y%m%d.%H%M%S')
    
    if mode=='SSM':
        
        if volumetric:
            [azi,ele]=np.meshgrid(azi,ele)
            azi=azi.ravel()
            ele=ele.ravel()
            vol_flag='vol.'
        else:
            vol_flag=''
            
        if isinstance(azi, np.ndarray):
            L=''
            for a,e in zip(azi[:-1],ele[:-1]):
                L=L+('%07.3f' % a+ '%07.3f' % e +'\n')
        else:
            L='%07.3f' % azi+ '%07.3f' % ele +'\n'
            
            
        save_name=f'{identifier}.{mode}.{vol_flag}.txt'
    
    os.makedirs(save_path,exist_ok=True)
    with open(os.path.join(save_path,save_name),'w') as fid:
        fid.write(L*repeats)
        fid.close()
        
                       