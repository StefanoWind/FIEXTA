# -*- coding: utf-8 -*-
"""
Lidar Statistical Barnes Objective Analysis
"""

from scipy.special import gamma
from typing import Union, Optional
from scipy.interpolate import interpn
import itertools
import numpy as np
from lisboa.utilities import mid, get_logger, with_logging, _load_configuration
from lisboa.config import LisboaConfig

class statistics:
    def __init__(self,
                 config: Union[str, dict, LisboaConfig],
                 verbose: bool = True,
                 logger: Optional[object] = None,
                 logfile: Optional[str] = None):
        
        self.logger = get_logger(verbose=verbose, logger=logger,filename=logfile)
        self.logger.log("Initializing LiSBOA")
    
        # Load configuration based on input type
        self.config,exit_flag = _load_configuration(config)
        self.logger.log(exit_flag)
        if self.config is None:
            return
        else:
            LisboaConfig.validate(self.config)
    
    @with_logging    
    def calculate_weights(
            self,
            x_exp: list,
            f=None):
        
        #reject non valid points
        n=len(self.config.Dn0) 
        if f is None:
            real=~np.isnan(np.sum(np.array(x_exp),axis=0))
        else:
            real=~np.isnan(np.sum(np.array(x_exp),axis=0)+f)
            
        for j in range(n):
            x_exp[j]=x_exp[j][real]

        f=f[real]
        
        #initialize variables
        Dn0=np.array(self.config.Dn0)   
        N=len(x_exp[0])
        x=np.zeros((n,N))
        xc=np.zeros(n)
        X_edg=[];
        X_cen=[];
        grid=[]
        
        self.logger.log(f"Calculating LiSBOA weights for {N} experimental points")
      
        #LiSBOA setup
        dx=self.config.grid_factor*Dn0
        r_max=self.config.r_max*self.config.sigma
        V=np.pi**(n/2)/gamma(n/2+1)*r_max**n
        
        #grid definition
        for j in range(n):
            xc[j]=np.mean(x_exp[j])#centroid
            x[j]=(x_exp[j]-xc[j])/Dn0[j]#normalized coordinate  
            X_edg.append((np.arange(self.config.mins[j]-dx[j]/2,self.config.maxs[j]+dx[j]*1.01,dx[j])-xc[j])/Dn0[j])#cell edges
            X_cen.append(mid(X_edg[j]))#cell center

        X=np.meshgrid(*[X_cen[j] for j in range(n)], indexing='ij')#broadcasting
        
        #grid in dimensional form
        for j in range(n):
            grid.append(X_cen[j]*Dn0[j]+xc[j])
            
        #zeroing
        w=np.zeros(np.shape(X[0]),dtype=object)
        sel=np.zeros(np.shape(X[0]),dtype=object)
        Dd=np.zeros(np.shape(X[0]))
        edge=Dd==1
        nodes=np.where(X[0])
        ctr=0
        
        #loop over all grid points
        for i in zip(*[xx for xx in nodes]):

            #squared Euclidean distance from obs points
            distSq=0
            for j in range(n):
                distSq+=(x[j]-X[j][i])**2
            
            #weights
            s=np.where(distSq<r_max**2)[0]   
            if len(s)>0:
                w[i]=np.exp(-distSq[s]/(2*self.config.sigma**2))
                
            #local spacing
            if Dd[i]!=10**99:   
                if len(s)>1: 
                    
                    #collapse points closer than tol_dist
                    pos_uni=np.round(x[:,s]/self.config.tol_dist)*self.config.tol_dist
                    N_uni= len(np.unique(pos_uni,axis=1)[0,:])
                    
                    #calculate local spacing assuming isotropy
                    if N_uni>1:
                        Dd[i]=V**(1/n)/(N_uni**(1/n)-1)
                    else:
                        Dd[i]=10**99
                else:
                    Dd[i]=10**99
                    
            #store
            sel[i]=s
            
            ctr+=1
            if int(ctr/100)==ctr/100:
                self.logger.log(f"{ctr}/{len(X[0].ravel())} grid points done")
                
        #find edge points and set spacing to infinity
        
        for i in zip(*[xx for xx in nodes]):
            ind_inf=[]
            if Dd[i]>self.config.max_Dd:
                for j in range(n):
                    i1=max(i[j]-self.config.dist_edge,0)
                    i2=min(i[j]+self.config.dist_edge,np.shape(X[0])[j]-1)
                    ind_inf.append(np.arange(i1,i2+1).astype(int))                
                for i_inf in itertools.product(*[ii for ii in ind_inf]):
                    edge[i_inf]=True
        Dd[edge]=10**9

        return grid,Dd,w,sel,x_exp,f
        
    def calculate_statistics(self,
                            x_exp: list,
                            f=None):
        
        grid,Dd,w,sel,x_exp,f=self.calculate_weights(x_exp,f)
        
        #zeroing
        WM=np.zeros(np.shape(Dd))+np.nan
        df=f
        nodes=np.where(Dd)
        
        excl=Dd>self.config.max_Dd#exclude undersampled region
        
        #iterations
        for m in range(self.config.max_iter):
            self.logger.log(f'Calculating statistics: iteration {m+1}/{self.config.max_iter}')
            for i in zip(*[xx for xx in nodes]):
                if not excl[i]:
                    fs=np.array(df[sel[i]])
                    ws=np.array(w[i])
                    reals=~np.isnan(fs+ws)
                    if np.sum(reals)>0:
                        fs=fs[reals]
                        ws=ws[reals]                     
                        WM[i]=np.sum(fs*ws)/np.sum(ws)                       
            if m==0:
                avg=WM.copy()
            else:
                avg+=WM
            
            #residual
            df=f-interpn(tuple(grid),avg,np.array(x_exp).T,bounds_error=False,fill_value=np.nan)
                
        return grid,Dd,excl,avg,None
   
        
# def li(x_exp,mins,maxs,Dn0,sigma,max_iter=None,calculate_stats=False,f=None,order=None,R_max=3,grid_factor=0.25,tol_dist=0.1,max_Dd=1):
#     #03/01/2021 (v 5): undersampling checkes in hypercubes instead of hyperspheres (faster)
#     #03/02/2022: finalized
#     from scipy.special import gamma
#     from scipy.interpolate import interpn
#     import itertools
#     import sys
#     import time
         
#     #outliers rejection
#     n=len(Dn0)    
#     if calculate_stats:
#         real=~np.isnan(np.sum(np.array(x_exp),axis=0)+f)
#         for j in range(n):
#             x_exp[j]=x_exp[j][real]
#         f=f[real]
#     else:
#         real=~np.isnan(np.sum(np.array(x_exp),axis=0))
#         for j in range(n):
#             x_exp[j]=x_exp[j][real]
    
#     #Initialization
#     t0=time.time()
#     Dn0=np.array(Dn0)   
#     N=len(x_exp[0])
#     x=np.zeros((n,N))
#     xc=np.zeros(n)
#     X_edg=[];
#     X_cen=[];
#     X2=[]
#     avg=None
  
    
#     #LiSBOA setup
#     dx=grid_factor*Dn0
#     R_max=R_max*sigma
#     V=np.pi**(n/2)/gamma(n/2+1)*R_max**n
#     precision=np.ceil(np.log10(np.max(np.abs(np.array(x_exp)/tol_dist))))+1
    
#     #define grid
#     for j in range(n):
#         xc[j]=np.min(x_exp)#centroid
#         x[j]=(x_exp[j]-xc[j])/Dn0[j]#normalized coordinate  
#         X_edg.append((np.arange(mins[j]-dx[j],maxs[j]+dx[j]*2,dx[j])-xc[j])/Dn0[j])#cell edges
#         X_cen.append(SL.mid(X_edg[j]))

#     X=np.meshgrid(*[X_cen[j] for j in range(n)], indexing='ij')
    
#     for j in range(n):
#         X2.append(X[j]*Dn0[j]+xc[j])
      
#     w=np.zeros(np.shape(X[0]),dtype=object)
#     sel=np.zeros(np.shape(X[0]),dtype=object)
#     val=np.zeros(np.shape(X[0]),dtype=object)
#     Dd=np.zeros(np.shape(X[0]))
#     N_grid=X[0].size
#     dist_inf=np.zeros(n)
    
#     #weights
#     for j in range(n):
#         dist_inf[j]=np.ceil(R_max/(dx[j]/Dn0[j]))
        
#     nodes=np.where(X[0])
#     ctr=0
#     for i in zip(*[xx for xx in nodes]):
#         distSq=0
#         for j in range(n):
#             distSq+=(x[j]-X[j][i])**2
#         s=np.where(distSq<R_max**2)   
#         if len(s)>0:
#             w[i]=np.exp(-distSq[s]/(2*sigma**2))
       
#         #local spacing
#         if Dd[i]!=10^99:   
#             if len(s[0])>1:                
#                 pos_uni=np.around(x[0][s]/tol_dist)*tol_dist
#                 for j in range(1,n):                
#                     pos_uni+=np.around(x[j][s]/tol_dist)*tol_dist*(10**precision)**j          
#                 N_uni= len(np.unique(np.array(pos_uni)))
                
#                 if N_uni>1:
#                     Dd[i]=V**(1/n)/(N_uni**(1/n)-1)
#                 else:
#                     Dd[i]=np.inf
#             else:
#                 Dd[i]=np.inf
                
#             ind_inf=[]
#             if Dd[i]>max_Dd:
#                 for j in range(n):
#                     i1=max(i[j]-dist_inf[j],0)
#                     i2=min(i[j]+dist_inf[j],np.shape(X[0])[j])
#                     ind_inf.append(np.arange(i1,i2).astype(int))                
#                 for i_inf in itertools.product(*[ii for ii in ind_inf]):
#                     Dd[i_inf]=10^99
#         #store
#         sel[i]=s
        
#         ctr+=1
#         if np.floor(ctr/N_grid*100)>np.floor((ctr-1)/N_grid*100):
#             est_time=(time.time()-t0)/ctr*(N_grid-ctr)
#             sys.stdout.write('\r LiSBOA:'+str(np.floor(ctr/N_grid*100).astype(int))+'% done, '+str(round(est_time))+' s left.') 
#     sys.stdout.write('\r                                                                         ')
#     sys.stdout.flush()
#     excl=Dd>max_Dd
                
#     #stats
#     if calculate_stats:
#         WM=np.zeros(np.shape(X[0]))+np.nan
#         avg=[]
#         df=f
#         for m in range(max_iter):
#             sys.stdout.write('\r Iteration #'+str(m))
#             for i in zip(*[xx for xx in nodes]):
#                 val[i]=f[s]
#                 if not excl[i]:
#                     fs=np.array(df[sel[i]])
#                     ws=np.array(w[i])
#                     reals=~np.isnan(fs+ws)
#                     if sum(reals)>0:
#                         fs=fs[reals]
#                         ws=ws[reals]                     
#                         WM[i]=sum(np.multiply(fs,ws))/sum(ws)                       
#             if m==0:
#                 avg.append(WM+0)
#             else:
#                 avg.append(avg[m-1]+WM)
            
#             df=f-interpn(tuple(X_cen),avg[m],np.transpose(x),bounds_error=False,fill_value=np.nan)

            
#     return X2,Dd,excl,avg