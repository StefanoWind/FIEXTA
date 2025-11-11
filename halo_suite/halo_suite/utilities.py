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
from scipy.optimize import brute
def scan_file_compiler(mode: str, 
                       azi: np.ndarray = np.array([]),
                       ele: np.ndarray = np.array([]),
                       azi_dir: np.ndarray = np.array([]),
                       ele_dir: np.ndarray = np.array([]),
                       repeats: int=1,
                       save_path: str='',
                       identifier: str='',
                       volumetric: bool=False,
                       optimize: bool=False,
                       reset: bool=False,
                       ppr: int=0,
                       S_azi: float=36,
                       S_ele: float=72,
                       A_azi: float=36,
                       A_ele: float=72,
                       S_min=0.1,
                       A_min=1,
                       config: dict={},
                       search_res: float=1):
    
    '''
    Writes scan text files for different scan mode:
        SSM: step and stare file based on the azimuths and elevations 
        CSM: continuous scan mode with two options:
            1) if speeds and accelrations are provided, it uses those kinematic parameters; 
               the azimuths and elevations are the start and end of the trajectory
            2) if config contains Dt_d, Dt_a, and Dt_p, it optimizes the kinematic parameters
               to match the angular steps provided in each segment
    '''
    
    #file system settings
    if save_path=='':
        save_path=os.path.join(cd,'scans')
    if identifier=='':
        identifier=datetime.strftime(datetime.now(),'%Y%m%d.%H%M%S')
        
    #array conversion for scalar inputs
    if type(azi).__name__!= 'ndarray':
        if type(azi).__name__!='list':
            azi=[azi]
        azi=np.array(azi)
    if type(ele).__name__!= 'ndarray':
        if type(ele).__name__!='list':
            ele=[ele]
        ele=np.array(ele)
    if type(azi_dir).__name__!= 'ndarray':
        if type(azi_dir).__name__!='list':
            azi_dir=[azi_dir]
        azi_dir=np.array(azi_dir)
    if type(ele_dir).__name__!= 'ndarray':
        if type(ele_dir).__name__!='list':
            ele_dir=[ele_dir]
        ele_dir=np.array(ele_dir)
        
    #clear nan
    azi=azi[~np.isnan(azi)]
    ele=ele[~np.isnan(ele)]
    azi_dir=azi_dir[~np.isnan(azi_dir)]
    ele_dir=ele_dir[~np.isnan(ele_dir)]
    
    #cartesian product of angles if it is a volumetric scan
    if volumetric:
        azi_dir=np.append(azi_dir,np.sign((azi[0] - azi[-1] + 180) % 360 - 180))
        ele_dir=np.append(1,ele_dir)
        
        [azi,ele]=np.meshgrid(azi,ele)
        azi=azi.ravel()
        ele=ele.ravel()
        
        [azi_dir,ele_dir]=np.meshgrid(azi_dir,ele_dir)
        azi_dir=azi_dir.ravel()[:-1]
        ele_dir=ele_dir.ravel()[:-1]
        vol_flag='vol.'
    else:
        vol_flag=''
    
    #add backswipe to home position
    if reset:
        if azi[0]!=azi[-1] or ele[0]!=ele[-1]:
            if len(azi_dir)!=0:
                azi_dir=np.append(azi_dir,np.sign((azi[0] - azi[-1] + 180) % 360 - 180))
                ele_dir=np.append(ele_dir,np.sign((ele[0] - ele[-1] + 180) % 360 - 180))
            
            azi=np.append(azi,azi[0])
            ele=np.append(ele,ele[0])
            
    #linearize angles
    azi=linearize_angle(azi, azi_dir)
    ele=linearize_angle(ele, ele_dir)
        
    #scan for SSM
    if mode=='SSM':
        L=''
        for a,e in zip(azi%360,ele%360):
            L=L+('%07.3f' % a+ '%07.3f' % e +'\n')
    
    if mode =='CSM':
        
        #extract step motors resolution
        ppd1=config['ppd_azi']
        ppd2=config['ppd_ele']
        
        #split scan into segments with same angular resolution
        dazi=(azi[1:] - azi[:-1] + 180) % 360 - 180
        dele=(ele[1:] - ele[:-1] + 180) % 360 - 180
        stop=np.concatenate(([0],np.where(np.abs((np.diff(dazi)+np.diff(dele)))>10**-10)[0]+1,[-1]))
        
        #program start-end points
        azi_range=azi[stop]
        P1=-azi_range*ppd1
        ele_range=ele[stop]
        P2=-ele_range*ppd2
        
        #do this if kinematic optimization is requested
        if optimize:
            
            #extract configs
            ang_tol=config['ang_tol']
            Dt_p=   config['Dt_p_CSM']
            Dt_a=   config['Dt_a_CSM']
            try:
                Dt_d=config['Dt_d_CSM'][ppr]
            except:
                Dt_d=0
            Dt_s=Dt_p+Dt_a*ppr
             
            #Halo units -> S.I.
            S_max_azi=config['S_max_azi']*10/ppd1
            A_max_azi=config['A_max_azi']*1000/ppd1
            S_max_ele=config['S_max_ele']*10/ppd2
            A_max_ele=config['A_max_ele']*1000/ppd2
            
            #zeroing
            S_azi=[S_max_azi]
            A_azi=[A_max_azi]
            S_ele=[S_max_ele]
            A_ele=[A_max_ele]
            
            #loop through segments
            ctr=0
            for i1,i2 in zip(stop[:-1],stop[1:]):
                dazi=(azi[i1+1]-azi[i1]+ 180) % 360 - 180
                dele=(ele[i1+1]-ele[i1]+ 180) % 360 - 180
                print(f'CSM scan optimization: {ctr}/{len(stop)-1} waypoints done.')    
                #azimuth step
                if np.abs(dazi)>ang_tol:#if angular resution exceeds the maximum one, go full speed
                    if np.abs(dazi)>S_max_azi*Dt_s:
                        S_azi=np.append(S_azi,S_max_azi)
                        A_azi=np.append(A_azi,A_max_azi)
                    else:#optimization of motor parameters to match median angular resolution
                        res = brute(angular_error,(slice(S_min,S_max_azi+search_res/2,search_res),
                                                   slice(A_min,A_max_azi+search_res/2,search_res)),
                                       args=(azi[i2]-azi[i1],dazi,ppr,Dt_p,Dt_a,Dt_d,ppd1),full_output=True, finish=True)
                        opt=res[0]
                        S_azi=np.append(S_azi,opt[0])
                        A_azi=np.append(A_azi,opt[1])
                else:
                    S_azi=np.append(S_azi,10)
                    A_azi=np.append(A_azi,10)
                
                #elevation step
                if np.abs(dele)>ang_tol:
                    if np.abs(dele)>S_max_ele*Dt_s:
                        S_ele=np.append(S_ele,S_max_ele)
                        A_ele=np.append(A_ele,A_max_ele)
                    else:
                        res = brute(angular_error,(slice(S_min,S_max_ele+search_res/2,search_res),
                                                   slice(A_min,A_max_ele+search_res/2,search_res)),
                                       args=(ele[i2]-ele[i1],dele,ppr,Dt_p,Dt_a,Dt_d,ppd2),full_output=True, finish=True)
                        opt=res[0]
                        S_ele=np.append(S_ele,opt[0])
                        A_ele=np.append(A_ele,opt[1])
                else:
                    S_ele=np.append(S_ele,10)
                    A_ele=np.append(A_ele,10)
                ctr+=1
                    
        #S.I. -> Halo units
        S1=S_azi*ppd1/10+np.zeros(len(P1))
        A1=A_azi*ppd1/1000+np.zeros(len(P1))
        S2=S_ele*ppd2/10+np.zeros(len(P2))
        A2=A_ele*ppd2/1000+np.zeros(len(P2))
        
        #write CSM file
        L=''
        for p1,p2,s1,s2,a1,a2 in zip(P1,P2,S1,S2,A1,A2):
            
            l='A.1=%.0f'%a1+',S.1=%.0f'%s1+',P.1=%.0f'%p1+'*A.2=%.0f'%a2+',S.2=%.0f'%s2+',P.2=%.0f'%p2+'\nW=0\n'
            L+=l
            
    #save file
    os.makedirs(save_path,exist_ok=True)
    save_name=f'{identifier}.{mode.lower()}.{vol_flag}txt'
    with open(os.path.join(save_path,save_name),'w') as fid:
        fid.write(L*repeats)
        fid.close()
    return os.path.join(save_path,save_name)
        
        
def read_hpl(file,config):
    '''
    Read kinematic from hlp lidar file
    '''
    
    #read ranges
    with open(file, "r") as fid:
        Nr=int(fid.readlines()[config['hpl_Nr']].split(':')[1].strip())
        fid.seek(0)
        dr=np.float64(fid.readlines()[config['hpl_dr']].split(':')[1].strip())
        fid.seek(0)
        ppr=int(fid.readlines()[config['hpl_ppr']].split(':')[1].strip())
        fid.seek(0)
        mode=fid.readlines()[config['hpl_mode']].split(' - ')[1].strip().upper()
        if mode=='STEPPED':
            mode='SSM'
        fid.seek(0)
        data=fid.readlines()[config['hpl_header']::Nr+1]
       

        #zeroing (specific file)
        tnum=[]
        azi=[]
        ele=[]
         
        #read only first gate info (time, angles)
        for d in np.array(data):
            d_split=np.array(d.split(' '))
            d_split=d_split[d_split!='']
            tnum=np.append(tnum,np.float64(d_split[config['hpl_index_time']])*3600)
            azi=np.append(azi,np.float64(d_split[config['hpl_index_azi']]))
            ele=np.append(ele,np.float64(d_split[config['hpl_index_ele']]))
                
        return tnum, azi, ele, Nr, dr, ppr, mode
        
def angular_error(params,ang_range,dang,ppr,Dt_p,Dt_a,Dt_d,ppd,ang_tol=10**-10):
    '''
    Squared error in median angular resolution for a given setup
    '''
    S=params[0]
    A=params[1]
    halo_sim=hls.halo_simulator(config={'processing_time': Dt_p,
                                        'acquisition_time':Dt_a,
                                        'dwell_time':      Dt_d,
                                        'ppd_azi':ppd,
                                        'ppd_ele':0})

    _,ang,_,_,_,_=halo_sim.scanning_head_sim(mode='CSM',ppr=ppr,azi=np.array([0,ang_range]),ele=np.array([0,0]),
                                                   S_azi=S,A_azi=A,S_ele=0,A_ele=0)
    
    #exclude points where the angle is dwelling 
    dang2=(ang[1:]-ang[:-1]+ 180) % 360 - 180
    if np.max(np.abs(dang2))>ang_tol:
        ang_error=np.sum((dang2[np.abs(dang2)>ang_tol]-dang)**2)
    else:
        ang_error=0
        
    return ang_error
        
def linearize_angle(ang,ang_dir):
    '''
    Make angle continuous by resolving circularity and preserving direction
    '''
    
    #wrap to 360
    ang=ang%360
    
    #optimize initial point
    if ang[0]>180: ang[0]-=360
    
    #calculate difference
    dang=(ang[1:] - ang[:-1] + 180) % 360 - 180
    
    #assign default direction if missing
    if len(ang_dir)==0:
        ang_dir=np.sign(dang)

    #fix points according to direction
    dang[(dang>0)*(ang_dir<0)]=dang[(dang>0)*(ang_dir<0)]-360
    dang[(dang<0)*(ang_dir>0)]=dang[(dang<0)*(ang_dir>0)]+360
    ang=ang[0]+np.cumsum(np.append(0,dang))
    
    return ang