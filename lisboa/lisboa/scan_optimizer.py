# -*- coding: utf-8 -*-
"""
Lidar Statistical Barnes Objective Analysis
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
from halo_suite import halo_simulator as hls 
from halo_suite.utilities import scan_file_compiler
from lisboa import statistics as stats
import lisboa.utilities as utl
from matplotlib import pyplot as plt
import xarray as xr
from datetime import datetime
import yaml

class scan_optimizer:
    def __init__(self,
                 config: dict,
                 save_path: str=os.path.join(cd,'data','Pareto'),
                 save_data: bool=True,
                 save_figures: bool=True,
                 identifier: str=''):
        
        if identifier!='':
            identifier='.'+identifier
        self.config=config
        self.save_figures=save_figures
        if save_data:
            self.save_name=os.path.join(save_path,datetime.strftime(datetime.now(),'%Y%m%d.%H%M')+identifier)
            os.makedirs(self.save_name)
      
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
        scan_file=np.empty((len(azi1),len(dazi)),dtype=object)
        
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
                
                scan_name=f'{a1:.2f}.{da:.2f}.{a2:.2f}.{e1:.2f}.{de:.2f}.{e2:.2f}'
                if mode=='SSM':
                    scan_file=scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=1,
                                                 identifier=scan_name,
                                                 volumetric=volumetric,reset=True)
                    
                    halo_sim=hls.halo_simulator(config={'processing_time':config_lidar['Dt_p_SSM'],
                                                         'acquisition_time':config_lidar['Dt_a_SSM'],
                                                         'dwell_time': 0})
                    
                    t,azi_sim,ele_sim,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode=mode,ppr=ppr,
                                                                                  S_azi=config_lidar['S_azi_SSM'],
                                                                                  A_azi=config_lidar['A_azi_SSM'],
                                                                                  S_ele=config_lidar['S_ele_SSM'],
                                                                                  A_ele=config_lidar['A_ele_SSM'],
                                                                                  source=scan_file)
                elif mode=='CSM':
                    scan_file=scan_file_compiler(mode=mode,azi=azi,ele=ele,repeats=1,ppr=ppr,
                                       identifier=scan_name,
                                       config=config_lidar,optimize=True,volumetric=volumetric,reset=True)
       
                    halo_sim=hls.halo_simulator(config={'processing_time':config_lidar['Dt_p_CSM'],
                                                        'acquisition_time':config_lidar['Dt_a_CSM'],
                                                        'dwell_time':config_lidar['Dt_d_CSM'][ppr],
                                                        'ppd_azi':config_lidar['ppd_azi'],
                                                        'ppd_ele':config_lidar['ppd_ele']})
                
                    t,azi_sim,ele_sim,t_all,azi_all,ele_all=halo_sim.scanning_head_sim(mode=mode,ppr=ppr,source=scan_file)
                    
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
                grid,Dd,excl,_,_,_,_=lproc.calculate_weights(x_exp)
                epsilon1[i_ang,i_dang]=np.sum(excl[i_ang,i_dang])/np.size(excl[i_ang,i_dang])
                
                #epsilon2
                L=np.floor(T/t[-1])
                p=np.arange(1,L)
                epsilon2[i_ang,i_dang]=(1/L+2/L**2*np.sum((L-p)*np.exp(-T/tau*p)))**0.5
                duration[i_ang,i_dang]=t[-1]
                
                #save data
                if hasattr(self,'save_name'):
                    
                    space_coords={}
                    ctr=0
                    for c in coords:
                        space_coords[c]=grid[ctr]
                        ctr+=1
                        
                    Output=xr.Dataset()
                    Output['Dd']=xr.DataArray(Dd,coords=space_coords,
                                        attrs={'description':'Local normalized data spacing'})
                    Output['excl']=xr.DataArray(excl,coords=space_coords,
                                        attrs={'description':'Grid points failing the Peterson-Middleton test'})
                    Output['x_points']=xr.DataArray(x.ravel(),coords={'index':np.arange(len(x.ravel()))},
                                        attrs={'description':'x of experimental points'})
                    Output['y_points']=xr.DataArray(y.ravel(),coords={'index':np.arange(len(y.ravel()))},
                                        attrs={'description':'y of experimental points'})
                    Output['z_points']=xr.DataArray(z.ravel(),coords={'index':np.arange(len(z.ravel()))},
                                        attrs={'description':'z of experimental points'})
                    Output['azi']=xr.DataArray(azi,coords={'index_azi':len(azi)},
                                               attrs={'description':'azimuth angle','units':'degrees'})
                    Output['ele']=xr.DataArray(azi,coords={'index_ele':len(ele)},
                                               attrs={'description':'elevation angle','units':'degrees'})
                    Output['r']=xr.DataArray(azi,coords={'index_r':len(r)},
                                               attrs={'description':'range','units':'m'})
                    
                    Output.attrs={}
                    for c in self.config:
                        Output.attrs[f'config_{c}']=self.config[c]
                    Output.attrs['scan_file']=scan_file
                    
                    Output.attrs['epsilon1']=epsilon1[i_ang,i_dang]
                    Output.attrs['epsilon2']=epsilon2[i_ang,i_dang]
                    Output.attrs['duration']=duration[i_ang,i_dang]
                    Output.to_netcdf(os.path.join(self.save_name,scan_name+'.nc'))
                    
                if self.save_figures:
                    self.visualize_scan(Output)
                    
                i_dang+=1
            i_ang+=1
            
        #build output
        Output=xr.Dataset()
        Output['epsilon1']=xr.DataArray(epsilon1,coords={'index_ang':np.arange(len(azi1)),'index_dang':np.arange(len(dazi))},
                            attrs={'description':'fraction of undersampled volume'})
        Output['epsilon2']=xr.DataArray(epsilon2,coords={'index_ang':np.arange(len(azi1)),'index_dang':np.arange(len(dazi))},
                            attrs={'description':'normalized error on the mean'})
        Output['duration']=xr.DataArray(duration,coords={'index_ang':np.arange(len(azi1)),'index_dang':np.arange(len(dazi))},
                            attrs={'description':'Scan duration'})
        Output['azi1']=xr.DataArray(azi1,coords={'index_ang':np.arange(len(azi1))},
                            attrs={'description':'start azimuth','units':'degrees'})
        Output['azi2']=xr.DataArray(azi2,coords={'index_ang':np.arange(len(azi1))},
                            attrs={'description':'end azimuth','units':'degrees'})
        Output['ele1']=xr.DataArray(ele1,coords={'index_ang':np.arange(len(azi1))},
                            attrs={'description':'start elevation','units':'degrees'})
        Output['ele2']=xr.DataArray(ele2,coords={'index_ang':np.arange(len(azi1))},
                            attrs={'description':'end elevation','units':'degrees'})
        Output['dazi']=xr.DataArray(dazi,coords={'index_dang':np.arange(len(dazi))},
                            attrs={'description':'azimuth step','units':'degrees'})
        Output['dele']=xr.DataArray(dele,coords={'index_dang':np.arange(len(dazi))},
                            attrs={'description':'end elevation','units':'degrees'})
        
        
        Output.to_netcdf(os.path.join(self.save_name+'.nc'))
        
        return Output
    
    def visualize_scan(self,Data):
        info=r'$\alpha='+str(Data.azi[0])+':'+str(Data.azi[-1])+':'+str(Data.ele.diff()[0])+'^\circ$'+ '\n'+\
             r'$\beta=' +str(Data.ele[0])+':'+str(Data.ele[-1])+':'+str(Data.ele.diff()[0])+'^\circ$' + '\n'+\
             r'$r= '    +str(Data.r[0])+':'+str(Data.r[-1])+':'+str(Data.r.diff()[0])+ '$ m' +'\n'+\
             r'$\Delta n_0='+str(self.config['Dn0'])+r'$ m'                                         + '\n'+\
             r'$\epsilon_I='+str(np.round(Data.attrs.epsilon1,2))+r'$'                      + '\n'+\
             r'$\epsilon_{II}='+str(np.round(Data.attrs.epsilon2,2))+r'$'                   + '\n'+\
             r'$\tau_s='+str(np.round(Data.attrs.duration,1))+r'$ s'
         
        coords=''
        for c in Data.excl.coords:
            coords+=c
   
        if coords!='xyz':
            fill=np.zeros(np.shape(Data.excl))
            fill[Data.excl==False]=10
            ax=plt.subplot(111)
            plt.pcolor(Data[coords[0]],Data[coords[1]],fill.T,cmap='Greys',vmin=0,vmax=1,alpha=0.5)
            plt.plot(Data[coords[0]+'_points'],Data[coords[1]+'_points'],'.k',markersize=2)
            ax.set_aspect('equal')
            ax.set_xlim([self.config['mins'][0],self.config['maxs'][0]])
            ax.set_ylim([self.config['mins'][1],self.config['maxs'][1]])
            ax.set_xlabel(r'$'+str(coords[0])+'$ [m]')
            ax.set_ylabel(r'$'+str(coords[1])+'$ [m]')
            dtick=np.max([np.diff(ax.get_xticks())[0],
                          np.diff(ax.get_yticks())[0]])
            ax.set_xticks(np.arange(self.config['mins'][0],self.config['maxs'][0]+dtick,dtick))
            ax.set_yticks(np.arange(self.config['mins'][1],self.config['maxs'][1]+dtick,dtick))
            ax.text(0,self.config['maxs'][1]/2,s=info,bbox={'edgecolor':'k','facecolor':(1,1,1,0.25)})

        elif coords=='xyz':
            fill=excl[i_ang,i_dang]==False
            ax=plt.subplot(N_row,N_col,i_dang+1,projection='3d')
            dx=np.diff(grid[i_ang,i_dang][0])[0]
            dy=np.diff(grid[i_ang,i_dang][1])[0]
            dz=np.diff(grid[i_ang,i_dang][2])[0]
            X,Y,Z=np.meshgrid(np.append(grid[i_ang,i_dang][0]-dx/2,grid[i_ang,i_dang][0][-1]+dx),
                      np.append(grid[i_ang,i_dang][1]-dy/2,grid[i_ang,i_dang][1][-1]+dy),
                      np.append(grid[i_ang,i_dang][2]-dz/2,grid[i_ang,i_dang][2][-1]+dz),indexing='ij')
            ax.voxels(X,Y,Z,fill,facecolor=colors[i_dang],alpha=0.1)
            sel_x=(points[i_ang,i_dang][0]>config['mins'][0])*(points[i_ang,i_dang][0]<config['maxs'][0])
            sel_y=(points[i_ang,i_dang][1]>config['mins'][1])*(points[i_ang,i_dang][1]<config['maxs'][1])
            sel_z=(points[i_ang,i_dang][2]>config['mins'][2])*(points[i_ang,i_dang][2]<config['maxs'][2])
            sel=sel_x*sel_y*sel_z
            plt.plot(points[i_ang,i_dang][0][sel],points[i_ang,i_dang][1][sel],points[i_ang,i_dang][2][sel],'.k',markersize=2)
            ax.set_aspect('equal')
            ax.set_xlim([config['mins'][0],config['maxs'][0]])
            ax.set_ylim([config['mins'][1],config['maxs'][1]])
            ax.set_zlim([config['mins'][2],config['maxs'][2]])
            ax.set_xlabel(r'$x$ [m]')
            ax.set_ylabel(r'$y$ [m]')
            ax.set_zlabel(r'$z$ [m]')
            dtick=np.max([np.diff(ax.get_xticks())[0],
                          np.diff(ax.get_yticks())[0],
                          np.diff(ax.get_zticks())[0]])
            ax.set_xticks(np.arange(config['mins'][0],config['maxs'][0]+dtick,dtick))
            ax.set_yticks(np.arange(config['mins'][1],config['maxs'][1]+dtick,dtick))
            ax.set_zticks(np.arange(config['mins'][2],config['maxs'][2]+dtick,dtick))
            ax.text(0,0,config['maxs'][2]/2,s=info,bbox={'edgecolor':'k','facecolor':(1,1,1,0.25)})
          
    def visualize_Pareto(self,Data):
        
        #extract information
      
        N_ang=len(Pareto.index_ang)
        N_dang=len(Pareto.index_dang)
        epsilon1=Pareto.epsilon1.values
        epsilon2=Pareto.epsilon2.values
        excl=Pareto.excl.values
        duration=Pareto.duration.values
        azi1=Pareto.azi1
        azi2=Pareto.azi2
        ele1=Pareto.ele1
        ele2=Pareto.ele2
        dazi=Pareto.dazi
        dele=Pareto.dele
        
        #plot Pareto front
        fig=plt.figure(figsize=(18,8))
        cmap = plt.cm.jet
        colors = [cmap(v) for v in np.linspace(0,1,len(Pareto.dazi))]
        N_row=int(np.floor(N_ang**0.5))
        N_col=int(np.ceil(N_ang/N_row))
        for i_ang in range(N_ang):
            ax=plt.subplot(N_row,N_col,i_ang+1)
            plt.plot(epsilon1[np.arange(N_ang)!=i_ang,:],epsilon2[np.arange(N_ang)!=i_ang,:],'.k',markersize=30,alpha=0.25)
            for i_dang in range(N_dang):
                plt.plot(epsilon1[i_ang,i_dang],epsilon2[i_ang,i_dang],'.',
                         color=colors[i_dang],markeredgecolor='k',markersize=30,
                         label=r'$\Delta \alpha='+str(dazi[i_dang])+r'^\circ$, $\Delta \beta='+str(dele[i_dang])+r'^\circ$')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xlabel(r'$\epsilon_I$')
            plt.ylabel(r'$\epsilon_{II}$')  
            plt.title(r'$\alpha \in ['+str(azi1[i_ang])+', '+str(azi2[i_ang])+r']^\circ$, $\beta \in ['+str(ele1[i_ang])+r', '+str(ele2[i_ang])+']^\circ$')
            plt.grid()
        plt.legend(draggable=True)

        #plot all scan geometries
        for i_ang in range(N_ang):
            fig=plt.figure(figsize=(18,8))
            N_row=int(np.floor(N_dang**0.5))
            N_col=int(np.ceil(N_dang/N_row))
            for i_dang in range(N_dang):  
                info=r'$\alpha='+str(azi1[i_ang])+':'+str(dazi[i_dang])+':'+str(azi2[i_ang])+'^\circ$'+ '\n'+\
                     r'$\beta=' +str(ele1[i_ang])+':'+str(dele[i_dang])+':'+str(ele2[i_ang])+'^\circ$' + '\n'+\
                     r'$\Delta n_0='+str(self.config['Dn0'])+r'$ m'                                         + '\n'+\
                     r'$\epsilon_I='+str(np.round(epsilon1[i_ang,i_dang],2))+r'$'                      + '\n'+\
                     r'$\epsilon_{II}='+str(np.round(epsilon2[i_ang,i_dang],2))+r'$'                   + '\n'+\
                     r'$\tau_s='+str(np.round(duration[i_ang,i_dang],1))+r'$ s'
            
                if coords!='xyz':
                    fill=np.zeros(np.shape(excl[i_ang,i_dang]))
                    fill[excl[i_ang,i_dang]==False]=10
                    
                    ax=plt.subplot(N_row,N_col,i_dang+1)
                    plt.pcolor(grid[i_ang,i_dang][0],grid[i_ang,i_dang][1],fill.T,cmap='Greys',vmin=0,vmax=1,alpha=0.5)
                    plt.plot(points[i_ang,i_dang][0],points[i_ang,i_dang][1],'.',color=colors[i_dang],markersize=2)
                    ax.set_aspect('equal')
                    ax.set_xlim([config['mins'][0],config['maxs'][0]])
                    ax.set_ylim([config['mins'][1],config['maxs'][1]])
                    ax.set_xlabel(r'$'+str(coords[0])+'$ [m]')
                    ax.set_ylabel(r'$'+str(coords[1])+'$ [m]')
                    dtick=np.max([np.diff(ax.get_xticks())[0],
                                  np.diff(ax.get_yticks())[0]])
                    ax.set_xticks(np.arange(config['mins'][0],config['maxs'][0]+dtick,dtick))
                    ax.set_yticks(np.arange(config['mins'][1],config['maxs'][1]+dtick,dtick))
                    ax.text(0,config['maxs'][1]/2,s=info,bbox={'edgecolor':'k','facecolor':(1,1,1,0.25)})

                elif coords=='xyz':
                    fill=excl[i_ang,i_dang]==False
                    ax=plt.subplot(N_row,N_col,i_dang+1,projection='3d')
                    dx=np.diff(grid[i_ang,i_dang][0])[0]
                    dy=np.diff(grid[i_ang,i_dang][1])[0]
                    dz=np.diff(grid[i_ang,i_dang][2])[0]
                    X,Y,Z=np.meshgrid(np.append(grid[i_ang,i_dang][0]-dx/2,grid[i_ang,i_dang][0][-1]+dx),
                              np.append(grid[i_ang,i_dang][1]-dy/2,grid[i_ang,i_dang][1][-1]+dy),
                              np.append(grid[i_ang,i_dang][2]-dz/2,grid[i_ang,i_dang][2][-1]+dz),indexing='ij')
                    ax.voxels(X,Y,Z,fill,facecolor=colors[i_dang],alpha=0.1)
                    sel_x=(points[i_ang,i_dang][0]>config['mins'][0])*(points[i_ang,i_dang][0]<config['maxs'][0])
                    sel_y=(points[i_ang,i_dang][1]>config['mins'][1])*(points[i_ang,i_dang][1]<config['maxs'][1])
                    sel_z=(points[i_ang,i_dang][2]>config['mins'][2])*(points[i_ang,i_dang][2]<config['maxs'][2])
                    sel=sel_x*sel_y*sel_z
                    plt.plot(points[i_ang,i_dang][0][sel],points[i_ang,i_dang][1][sel],points[i_ang,i_dang][2][sel],'.k',markersize=2)
                    ax.set_aspect('equal')
                    ax.set_xlim([config['mins'][0],config['maxs'][0]])
                    ax.set_ylim([config['mins'][1],config['maxs'][1]])
                    ax.set_zlim([config['mins'][2],config['maxs'][2]])
                    ax.set_xlabel(r'$x$ [m]')
                    ax.set_ylabel(r'$y$ [m]')
                    ax.set_zlabel(r'$z$ [m]')
                    dtick=np.max([np.diff(ax.get_xticks())[0],
                                  np.diff(ax.get_yticks())[0],
                                  np.diff(ax.get_zticks())[0]])
                    ax.set_xticks(np.arange(config['mins'][0],config['maxs'][0]+dtick,dtick))
                    ax.set_yticks(np.arange(config['mins'][1],config['maxs'][1]+dtick,dtick))
                    ax.set_zticks(np.arange(config['mins'][2],config['maxs'][2]+dtick,dtick))
                    ax.text(0,0,config['maxs'][2]/2,s=info,bbox={'edgecolor':'k','facecolor':(1,1,1,0.25)})


def stack(array):
    
    return np.stack([[array[i, j] for j in range(array.shape[1])] for i in range(array.shape[0])])
                     