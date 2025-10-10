# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 14:21:06 2025

@author: sletizia
"""
import numpy as np
import os
import re
class halo_simulator:
    '''
    Simulator of the movement of the lidar scanninf head
    '''
    def __init__(self,
                 config: dict,
                 source_scan: str=''):
    
        self.config=config
    
    def scanning_head_sim(self,
                          mode: str='SSM',
                          ppr: int=1000,
                          azi: np.array=np.array([0,0]),
                          ele: np.array=np.array([0,0]),
                          source: str='',
                          S_azi: float=21,
                          S_ele: float=72,
                          A_azi: float=40,
                          A_ele: float=40,
                          dt: float=0.01,
                          ang_tol: float=0.1,
                          azi0: float=0,
                          ele0: float=0):
        '''
        This module takes as input either:
            1) for SSM, a sequence of azimuths and elevations and maximum speeds and accelrations
            2) for CSM, either a txt scan file or specific valies of azimuths, elevations, speeds and accelerations
        '''
        #extract config
        Dt_p=self.config['processing_time']
        Dt_a=self.config['acquisition_time']
        Dt_d=self.config['dwell_time']
        Dt_s=Dt_p+Dt_a*ppr
        if mode=='csm':
            ppd1=self.config['ppd_azi']
            ppd2=self.config['ppd_ele']
        
        #read scan file if provided
        if os.path.isfile(source) and mode=='csm':
            with open(source,'r') as fid:
                lines = fid.readlines()
            lines = [line.strip() for line in lines]
            
            #extract kinematic parameters
            P1=[]
            P2=[]
            S1=[]
            S2=[]
            A1=[]
            A2=[]
            for l in lines:
                if len(l)>10:
                    kin = [np.float64(v[1:]) for v in re.findall(r'=[-]?\d*\.?\d+', l)]
                
                    A1=np.append(A1,kin[0])
                    S1=np.append(S1,kin[1])
                    P1=np.append(P1,kin[2])
                    A2=np.append(A2,kin[3])
                    S2=np.append(S2,kin[4])
                    P2=np.append(P2,kin[5])
                    
            azi=np.append(azi0,-np.round(P1/ppd1/ang_tol)*ang_tol)
            ele=np.append(ele0,-np.round(P2/ppd2/ang_tol)*ang_tol)
            S_azi=S1*10/ppd1
            S_ele=S2*10/ppd2
            A_azi=A1*1000/ppd1
            A_ele=A2*1000/ppd2
        else:
            azi=np.append(azi0,azi)
            ele=np.append(ele0,ele)
                
        #wrap angles
        azi=azi%360
        ele=ele%360
        
        if mode=='ssm':
            #for SSM, use the maximum speeds and acceleration provided in the config
            S_azi=self.config['max_S_azi']+np.zeros((len(azi))-1)
            S_ele=self.config['max_S_ele']+np.zeros((len(azi))-1)
            A_azi=self.config['max_A_azi']+np.zeros((len(azi))-1)
            A_ele=self.config['max_A_ele']+np.zeros((len(azi))-1)
            
        #zeroing
        azi_all=np.array([azi[0]])
        ele_all=np.array([ele[0]])
        t_all=np.array([0])
        t=np.array([0])
        
        S_azi+=np.zeros((len(azi))-1)
        S_ele+=np.zeros((len(azi))-1)
        A_azi+=np.zeros((len(azi))-1)
        A_ele+=np.zeros((len(azi))-1)
        
        #loop through segments
        for azi1,azi2,ele1,ele2,S1,S2,A1,A2 in zip(azi[:-1],azi[1:],ele[:-1],ele[1:],S_azi,S_ele,A_azi,A_ele):
            dazi=((azi2 - azi1 + 180) % 360) - 180
            dele=((ele2 - ele1 + 180) % 360) - 180
            if np.abs(dazi)>ang_tol and np.abs(dele)<ang_tol:#PPI simulation
                _t,_azi=self.step_scanning_head(azi1,azi2,S1,A1,mode=mode)
                _ele=ele1+_azi*0
            elif np.abs(dazi)<ang_tol and np.abs(dele)>ang_tol:#RHI simulations
                _t,_ele=self.step_scanning_head(ele1,ele2,S2,A2,mode=mode)
                _azi=azi1+_ele*0
            elif np.abs(dazi)<ang_tol and np.abs(dele)<ang_tol:#stare
                    _t=[0,0]
                    _azi=[azi1,azi2]
                    _ele=[ele1,ele2]
            else:
                raise ValueError('The movement of azimuth and elevation within the same angular step is not supported.')
                
            azi_all=np.append(azi_all,_azi[1:])
            ele_all=np.append(ele_all,_ele[1:])
            if mode=='ssm':
                #in SSM, add acquisition delay
                t_all=np.append(t_all,t_all[-1]+_t[1:]+Dt_s)
                t=np.append(t,t[-1]+_t[-1]+Dt_s)
            elif mode=='csm':
                #in CSM mode, add the dwelling time
                t_all=np.append(t_all,t_all[-1]+_t[1:]+Dt_d)
        
        if mode=='csm':
            #in CSM mode, iterpolate at sampling point
            t=np.arange(0,t_all[-1]+Dt_s,Dt_s)
            azi=np.interp(t,t_all,azi_all)
            ele=np.interp(t,t_all,ele_all)
            
        return t,azi,ele,t_all,azi_all,ele_all
    
    def step_scanning_head(self,ang1,ang2,S,A,mode='ssm',dt=0.01):
        
        #fix 0 values
        if A==0:
            A+=10**-10
        if S==0:
            S+=10**-10
        
        #if SSM mode is on and path crosses 0 degrees, find shortest route
        if mode=='ssm':
            if ang2-ang1<-180:
                ang1-=360
            elif ang2-ang1>180:
                   ang2-=360     
        
        sign=np.sign(ang2-ang1)
        
        if np.abs(ang2-ang1)<S**2/A:#for angular ranges shorter than the critical value
            Dt_m=2*(np.abs(ang2-ang1)/A)**0.5
            t=np.arange(0,Dt_m+dt,dt)
            ang=np.zeros(len(t))
            t1=Dt_m/2
            ang[t<t1]=ang1+0.5*A*(t[t<t1])**2*sign
            ang[t>=t1]=ang1+(ang2-ang1)/2+A*t1*(t[t>=t1]-t1)*sign-0.5*A*(t[t>=t1]-t1)**2*sign
        else:#for angular ranges larger than the critical value
            Dt_m=np.abs(ang2-ang1)/S+S/A
            t=np.arange(0,Dt_m+dt,dt)
            ang=np.zeros(len(t))
            t1=S/A
            t2=Dt_m-S/A
            ang[t<t1]=ang1+0.5*A*(t[t<t1])**2*sign
            ang[(t>=t1)*(t<t2)]=ang1+S**2/(2*A)*sign+S*(t[(t>=t1)*(t<t2)]-t1)*sign
            ang[t>=t2]=ang2-S**2/A/2*sign+S*(t[t>=t2]-t2)*sign-0.5*A*(t[t>=t2]-t2)**2*sign
            
        return t,ang%360
        
        
        