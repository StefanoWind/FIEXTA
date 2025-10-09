# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 14:21:06 2025

@author: sletizia
"""
import numpy as np
class halo_simulator:
    def __init__(self,
                 config: dict,
                 source_scan: str=''):
    
        self.config=config
    
    def scanning_head_sim(self,
                          mode: str='SSM',
                          ppr: int=1000,
                          azi: np.array=np.array([0,0]),
                          ele: np.array=np.array([0,0]),
                          S_azi: float=21,
                          S_ele: float=72,
                          A_azi: float=40,
                          A_ele: float=40,
                          dt: float=0.01,
                          ang_tol: float=0.1,
                          azi0: float=0,
                          ele0: float=0):
        
        T_p=self.config['processing_time']
        T_a=self.config['acquisition_time']
        T_s=T_p+T_a*ppr
        
        azi=azi%360
        ele=ele%360
        
        if mode=='ssm':
            S_azi=self.config['max_S_azi']
            S_ele=self.config['max_S_ele']
            A_azi=self.config['max_A_azi']
            A_ele=self.config['max_A_ele']

        azi_all=np.array([azi[0]])
        ele_all=np.array([ele[0]])
        t_all=np.array([0])
        T_all=np.array([0])
        
        for azi1,azi2,ele1,ele2 in zip(azi[:-1],azi[1:],ele[:-1],ele[1:]):
            dazi=((azi2 - azi1 + 180) % 360) - 180
            dele=((ele2 - ele1 + 180) % 360) - 180
            if np.abs(dazi)>ang_tol and np.abs(dele)<ang_tol:
                t,_azi=self.step_scanning_head(azi1,azi2,S_azi,A_azi)
                _ele=ele1+_azi*0
            elif np.abs(dazi)<ang_tol and np.abs(dele)>ang_tol:
                t,_ele=self.step_scanning_head(ele1,ele2,S_ele,A_ele)
                _azi=azi1+_ele*0
            elif np.abs(dazi)<ang_tol and np.abs(dele)<ang_tol:
                    t=[0,0]
                    _azi=[azi1,azi2]
                    _ele=[ele1,ele2]
            else:
                raise ValueError('The movement of azimuth and elevation within the same angular step is not supported.')
                
            azi_all=np.append(azi_all,_azi[1:])
            ele_all=np.append(ele_all,_ele[1:])
            if mode=='ssm':
                t_all=np.append(t_all,t_all[-1]+t[1:]+T_s)
                T_all=np.append(T_all,T_all[-1]+t[-1]+T_s)
            elif mode=='csm':
                t_all=np.append(t_all,t_all[-1]+t[1:])
        
        if mode=='csm':
            T_all=np.arange(0,t[-1]+T_s,T_s)
            azi=np.interp(T_all,t_all,azi_all)
            ele=np.interp(T_all,t_all,ele_all)
            
        return T_all,t_all,azi_all,ele_all,azi,ele
    
    def step_scanning_head(self,ang1,ang2,S_max,A,dt=0.01):
        if ang2-ang1<-180:
            ang1-=360
        elif ang2-ang1>180:
               ang2-=360     
        sign=np.sign(ang2-ang1)
        if np.abs(ang2-ang1)<S_max**2/A:
            T=2*(np.abs(ang2-ang1)/A)**0.5
            t=np.arange(0,T+dt,dt)
            ang=np.zeros(len(t))
            t1=T/2
            ang[t<t1]=ang1+0.5*A*(t[t<t1])**2*sign
            ang[t>=t1]=ang1+(ang2-ang1)/2+A*t1*(t[t>=t1]-t1)*sign-0.5*A*(t[t>=t1]-t1)**2*sign
        else:
            T=np.abs(ang2-ang1)/S_max+S_max/A
            t=np.arange(0,T+dt,dt)
            ang=np.zeros(len(t))
            t1=S_max/A
            t2=T-S_max/A
            ang[t<t1]=ang1+0.5*A*(t[t<t1])**2*sign
            ang[(t>=t1)*(t<t2)]=ang1+S_max**2/(2*A)*sign+S_max*(t[(t>=t1)*(t<t2)]-t1)*sign
            ang[t>=t2]=ang2-S_max**2/A/2*sign+S_max*(t[t>=t2]-t2)*sign-0.5*A*(t[t>=t2]-t2)**2*sign
            
        return t,ang%360
        
        
        