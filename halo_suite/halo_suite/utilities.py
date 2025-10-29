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
                       config: dict={}):
    
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
            Dt_p=config['Dt_p_CSM']
            Dt_a=config['Dt_a_CSM']
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
            for i1,i2 in zip(stop[:-1],stop[1:]):
                dazi=(azi[i1+1]-azi[i1]+ 180) % 360 - 180
                dele=(ele[i1+1]-ele[i1]+ 180) % 360 - 180
                
                #cap maximum angular resolution
                if np.abs(dazi)>S_max_azi*Dt_s:
                    S_min_azi=S_max_azi-1
                    A_min_azi=A_max_azi-1
                else:
                    S_min_azi=ang_tol
                    A_min_azi=ang_tol
                if np.abs(dele)>S_max_ele*Dt_s:
                    dele=np.sign(dele)*(S_max_ele-1)*Dt_s
                    S_min_ele=S_max_ele-1
                    A_min_ele=A_max_ele-1
                else:
                    S_min_ele=ang_tol
                    A_min_ele=ang_tol
                    
                #optimization of motor paramters to match median angular resolution
                res = minimize(angular_error,[np.min([np.abs(dazi)/Dt_s,S_max_azi-0.5]),A_max_azi-0.5,
                                              np.min([np.abs(dele)/Dt_s,S_max_ele-0.5]),A_max_ele-0.5],
                               args=(azi[i2]-azi[i1],ele[i2]-ele[i1],dazi,dele,ppr,Dt_p,Dt_a,Dt_d,ppd1,ppd2),
                               bounds= [(S_min_azi, S_max_azi), (A_min_azi, A_max_azi), 
                                        (S_min_ele, S_max_ele), (A_min_ele, A_max_ele)])
                if res.success==False:
                    raise BaseException(f'Optimization of motion failed for azimuth step={dazi} deg, elevation step={dele} deg, PPR={ppr}.')
                opt=res.x
                
                #S.I. -> Halo units
                S_azi=np.append(S_azi,opt[0])
                A_azi=np.append(A_azi,opt[1])
                S_ele=np.append(S_ele,opt[2])
                A_ele=np.append(A_ele,opt[3])
           
        #convert to Halo's units
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
            

def angular_error(params,azi_range,ele_range,dazi,dele,ppr,Dt_p,Dt_a,Dt_d,ppd1,ppd2,ang_tol=10**-10):
    '''
    Squared error in median angular resolution for a given setup
    '''
    S_azi=params[0]
    A_azi=params[1]
    S_ele=params[2]
    A_ele=params[3]
    halo_sim=hls.halo_simulator(config={'processing_time': Dt_p,
                                        'acquisition_time':Dt_a,
                                        'dwell_time':      Dt_d,
                                        'ppd_azi':ppd1,
                                        'ppd_ele':ppd2})

    _,azi,ele,_,_,_=halo_sim.scanning_head_sim(mode='CSM',ppr=ppr,azi=np.array([0,azi_range]),ele=np.array([0,ele_range]),
                                                   S_azi=S_azi,A_azi=A_azi,S_ele=S_ele,A_ele=A_ele)
    
    #exclude points where the angle is dwelling 
    dazi2=(azi[1:]-azi[:-1]+ 180) % 360 - 180
    if np.max(np.abs(dazi2))>ang_tol:
        dazi2=np.append(dazi2[0],dazi2)
        azi_error=(np.median(np.diff(azi[np.abs(dazi2)>ang_tol]))-dazi)**2
    else:
        azi_error=0
        
    dele2=(ele[1:]-ele[:-1]+ 180) % 360 - 180
    
    if np.max(np.abs(dele2))>ang_tol:
        dele2=np.append(dele2[0],dele2)
        ele_error=(np.median(np.diff(ele[np.abs(dele2)>ang_tol]))-dele)**2
    else:
        ele_error=0
    
    return azi_error+ele_error
    
        
def linearize_angle(ang,ang_dir):
    '''
    Make angle continuous by resolving circularity and preserving direction
    '''
    
    #wrap to 360
    ang=ang%360
    
    #optimize initial point
    if ang[0]>180: ang[0]-=360
    
    #calculate smallest difference
    dang=(ang[1:] - ang[:-1] + 180) % 360 - 180
    
    #assign default direction if missing
    if len(ang_dir)==0:
        ang_dir=np.sign(dang)

    #fix points according to direction
    dang[(dang>0)*(ang_dir<0)]=dang[(dang>0)*(ang_dir<0)]-360
    dang[(dang<0)*(ang_dir>0)]=dang[(dang<0)*(ang_dir>0)]+360
    ang=ang[0]+np.cumsum(np.append(0,dang))
    
    return ang