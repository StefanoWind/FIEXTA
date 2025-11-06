# -*- coding: utf-8 -*-
"""
Scan optimizer using Pareto front based on LiSBOA results
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
                 make_figures: bool=True,
                 save_figures: bool=True,
                 identifier: str=''):
        
        if identifier!='':
            identifier='.'+identifier
        self.config=config
        self.make_figures=make_figures
        self.save_figures=save_figures
        if save_data:
            self.save_name=os.path.join(save_path,datetime.strftime(datetime.now(),'%Y%m%d.%H%M%S')+identifier)
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
        duration=np.zeros((len(azi1),len(dazi)))+np.nan
        
        #initialize LiSBOA
        lproc=stats.statistics(self.config)
        
        with open(path_config_lidar, 'r') as fid:
            config_lidar = yaml.safe_load(fid)    
        
        #scan testing
        r=np.arange(rmin,rmax+dr/2,dr)
        i_ang=0
        for a1,a2,e1,e2 in zip(azi1,azi2,ele1,ele2):#loop through angular sectors
            i_dang=0
            for da,de in zip(dazi,dele):#loop through angular resolutions
                print(f"Evaluating azi={a1}:{da}:{a2}, ele={e1}:{de}:{e2}")
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
                    
                    halo_sim=hls.halo_simulator(config={'processing_time':  config_lidar['Dt_p_SSM'],
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
                                       identifier=scan_name,config=config_lidar,
                                       optimize=True,volumetric=volumetric,reset=True)
       
                    halo_sim=hls.halo_simulator(config={'processing_time': config_lidar['Dt_p_CSM'],
                                                        'acquisition_time':config_lidar['Dt_a_CSM'],
                                                        'dwell_time':      config_lidar['Dt_d_CSM'][ppr],
                                                        'ppd_azi':         config_lidar['ppd_azi'],
                                                        'ppd_ele':         config_lidar['ppd_ele']})
                
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
                epsilon1[i_ang,i_dang]=np.sum(excl)/np.size(excl)
                
                #epsilon2
                if T>t[-1]:
                    L=np.floor(T/t[-1])
                    p=np.arange(1,L)
                    epsilon2[i_ang,i_dang]=(1/L+2/L**2*np.sum((L-p)*np.exp(-t[-1]/tau*p)))**0.5
                else:
                    epsilon2[i_ang,i_dang]=np.nan
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
                    Output['azi']=xr.DataArray(azi,coords={'index_azi':np.arange(len(azi))},
                                               attrs={'description':'azimuth angle','units':'degrees'})
                    Output['ele']=xr.DataArray(ele,coords={'index_ele':np.arange(len(ele))},
                                               attrs={'description':'elevation angle','units':'degrees'})
                    Output['r']=xr.DataArray(r,coords={'index_r':np.arange(len(r))},
                                               attrs={'description':'range','units':'m'})
                    
                    Output.attrs={}
                    for c in self.config:
                        Output.attrs[f'config_{c}']=self.config[c]
                    Output.attrs['scan_file']=scan_file
                    
                    Output.attrs['epsilon1']=epsilon1[i_ang,i_dang]
                    Output.attrs['epsilon2']=epsilon2[i_ang,i_dang]
                    Output.attrs['duration']=duration[i_ang,i_dang]
                    Output.attrs['volumetric']=str(volumetric)
                    Output.attrs['mode']=mode
                    Output.to_netcdf(os.path.join(self.save_name,scan_name+'.nc'))
                    
                if self.make_figures:
                    fig=utl.visualize_scan(Output)
                if self.save_figures:
                    fig.savefig(os.path.join(self.save_name,scan_name+'.png'))
                    plt.close(fig)
                    
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
        
        Output.to_netcdf(os.path.join(self.save_name,os.path.basename(self.save_name)+'.Pareto.nc'))
        
        if self.make_figures:
            fig=utl.visualize_Pareto(Output)
        if self.save_figures:
            fig.savefig(os.path.join(self.save_name,os.path.basename(self.save_name)+'.Pareto.png'))
        
        print(f'Pareto results saved as {os.path.basename(self.save_name)+".Pareto.nc"}')
            
        return Output
                     