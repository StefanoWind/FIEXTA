import os
import xarray as xr
import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
import re
import matplotlib
from matplotlib.patches import Circle
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
warnings.filterwarnings('ignore',message='Mean of empty slice')
warnings.filterwarnings('ignore',message='.*pcolor.*')
warnings.filterwarnings('ignore',message='.*encountered in log10.*')
warnings.filterwarnings('ignore',message='.*converting a masked element to nan.*')

class LIDARGO:
    def __init__(self, source, config_file, verbose=True,logger=None):
        '''
        Initialize the LIDAR data processor with configuration parameters.

        Inputs:
        ------
        source: str
            The input-level file name 
        config_file: str
            The configuration file
        verbose: bool
            Whether or not print LiSBOA output
        '''
        
        plt.close('all')
        
        self.source=source
        self.verbose = verbose
        self.logger=logger
        
        #load configuration
        configs=pd.read_excel(config_file).set_index('PARAMETER')
        matches=[]
        for regex in configs.columns:
            match = re.findall(regex, source)
            if len(match)>0:
                matches.append(regex)
        
        if len(matches)==0:
            self.print_and_log('No regular expression matching the file name')
            return
        elif len(matches)>1:
            self.print_and_log('Mulitple regular expressions matching the file name')
            return
                
        config=configs[matches[0]].to_dict()
        self.config = config
        
        # Dynamically assign attributes from the dictionary
        for key, value in config.items():
            setattr(self, key, value)
        
        #load data
        self.inputData = xr.open_dataset(self.source)
    
    def print_and_log(self,message):
        print(message)
        if self.logger is not None:
            self.logger.info(message)
            
    def process_scan(
        self, make_figures=True, save_file=False, save_path=None, replace=True, process_time=True
    ):
        """
        Run all the processing.
    
        Inputs:
        -------
        make_figures: bool
            Whether or not generate QC figures
        save_file: bool
            Whether or not save the final processed file
        save_path: str
            String with the directory the destination of the processed files.
            Optional, defaults to analogous input, replacing input-level with output-level
            data. Creates the necessary intermediate directories
        replace: bool
            Whether or not to replace processed scan if one already exists
    
        """
        
        # Check if file has been processed yet and whether is to be replaced
        if 'config' not in dir(self):
            self.print_and_log(f'No configuration available. Skipping file {self.source}')
            return
            
        #Compose filename
        save_filename = self.data_level_out.join(self.source.split(self.data_level_in))
        if save_path is not None:
            save_filename=os.path.join(save_path,os.path.basename(save_filename))
        
        self.save_filename=save_filename
        
        if os.path.exists(os.path.dirname(save_filename))==False:
            os.makedirs(os.path.dirname(save_filename))
            
        if save_file and not replace and os.path.isfile(save_filename):
            self.print_and_log(f'Processed file {save_filename} exists. Skipping it.')
            return
        else:
            self.print_and_log(f'Processing file {save_filename}.')
    
        self.statistics()
        
        if make_figures:    
            self.plots()
        
        if save_file:
            self.Data.to_netcdf(save_filename)
            
            
    def statistics(self):
        '''
        Calculate mean and standard deviation of de-projected wind speed through LiSBOA (Letizia et al., AMT, 2021)

        '''
        
        #LiSBOA settings
        D=self.diameter
        mins=[self.xmin,self.ymin,self.zmin]
        maxs=[self.xmax,self.ymax,self.zmax]
        Dn0=[self.Dn0_x,self.Dn0_y,self.Dn0_z]
        max_iter=self.max_iter
        sigma=self.sigma
        
        #Space-time coordinates
        time=self.inputData['time'].values
        x_lid=self.inputData['x'].values
        y_lid=self.inputData['y'].values
        z_lid=self.inputData['z'].values
        coords=[x_lid.ravel()/D,y_lid.ravel()/D,z_lid.ravel()/D]
        
        #Velocity de-projection (assumes 0 yaw misalignment)
        u_qc=(self.inputData['wind_speed'].where(self.inputData['qc_wind_speed']==0)/(self.inputData['x']/(self.inputData['x']**2+self.inputData['y']**2+self.inputData['z']**2)**0.5)).values

        #Run LiSBOA
        X2,Dd,excl,avg,HOM=LiSBOA_v7_2(coords,mins,maxs,Dn0,sigma,max_iter=max_iter,calculate_stats=True,f=u_qc.ravel(),order=2,R_max=3,grid_factor=0.25,tol_dist=0.1,max_Dd=1,verbose=self.verbose)
            
        #Extract information
        x=np.round(X2[0][:,0,0]*D,1)
        y=np.round(X2[1][0,:,0]*D,1)
        z=np.round(X2[2][0,0,:]*D,1)
        u_avg=avg[-1]
        u_avg[excl]=np.nan
        u_std=HOM[-1]**0.5
        u_std[excl]=np.nan
        
        self.Data=xr.Dataset()
        self.Data['u_avg']=xr.DataArray(data=u_avg,coords={'x':x,'y':y,'z':z},
                               attrs={'long_name':'Mean streamwise velocity',
                                      'units':'m/s',
                                      'description':'LiSBOA-average of the de-projected line-of-sight velocity assuming 0 yaw misalignment.'})
        self.Data['u_stdev']=xr.DataArray(data=u_std,coords={'x':x,'y':y,'z':z},
                               attrs={'long_name':'Standard deviation of streamwise velocity',
                                      'units':'m/s',
                                      'description':'LiSBOA-standard deviation of the de-projected line-of-sight velocity assuming 0 yaw misalignment.'})
        self.Data.attrs['start_time']=datestr(dt64_to_num(np.nanmin(time)),'%Y%m%d %H-%M-%S')
        self.Data.attrs['end_time']=  datestr(dt64_to_num(np.nanmax(time)),'%Y%m%d %H-%M-%S')
        
    def plots(self):
        '''
        Make figures.
        '''
        
        #Spatial coordinates
        x=self.Data['x'].values
        y=self.Data['y'].values
        z=self.Data['z'].values
        D=self.diameter
        H=self.ground_level
        
        #Extract hub-height plane
        u_avg_int=self.Data['u_avg'].sel(z=0).values
        u_std_int=self.Data['u_stdev'].sel(z=0).values
        TI_int=u_std_int.T/np.abs(u_avg_int.T)*100
        
        #Plot mean velocity at hub height
        fig=plt.figure(figsize=(18,6.6))
        ax = plt.subplot(1,2,1)
        ax.set_facecolor((0,0,0,0.2))
        levels=np.unique(np.round(np.linspace(np.nanpercentile(u_avg_int,5)-0.5, np.nanpercentile(u_avg_int,95)+0.5, 20),1))
        cf=plt.contourf(x,y,u_avg_int.T,levels, cmap='coolwarm',extend='both')
        plt.xlim([self.xmin*D,self.xmax*D])
        plt.ylim([self.ymin*D,self.ymax*D])
        plt.grid(alpha=0.5)
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$y$ [m]')
        plt.title('Mean streamwise velocity at on '+self.Data.attrs['start_time'][:8]+'\n File: '+os.path.basename(self.source)\
                  +'\n'+self.Data.attrs['start_time'][9:]+' - '+self.Data.attrs['end_time'][9:])
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.4)
        cax=fig.add_axes([ax.get_position().x0+ax.get_position().width+0.01,ax.get_position().y0,0.015,ax.get_position().height])
        plt.colorbar(cf, cax=cax,label=r'Mean streamwise velocity [m s$^{-1}$]')
        
        #plot TI at hub height
        ax = plt.subplot(1,2,2)
        ax.set_facecolor((0,0,0,0.2))
        cf=plt.contourf(x,y,TI_int,np.unique(np.round(np.linspace(np.nanpercentile(TI_int,5)-0.5, np.nanpercentile(TI_int,95)+0.5, 20),1)), cmap='hot',extend='both')
        plt.xlim([self.xmin*D,self.xmax*D])
        plt.ylim([self.ymin*D,self.ymax*D])
        plt.grid(alpha=0.5)
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$y$ [m]')
        plt.title('Turbulence intensity at on '+self.Data.attrs['start_time'][:8]+'\n File: '+os.path.basename(self.source)\
                  +'\n'+self.Data.attrs['start_time'][9:]+' - '+self.Data.attrs['end_time'][9:])
        cax=fig.add_axes([ax.get_position().x0+ax.get_position().width+0.01,ax.get_position().y0,0.015,ax.get_position().height])
        plt.colorbar(cf, cax=cax,label=r'Turbulence intensity [%]')
        
        fig.savefig(self.save_filename.replace('.nc','_hub_height.png'))
        plt.close()
        
        #y-z planes (for 3D scans only)
        if len(z)>1:
            x_plot_wake=[np.float(xp) for xp in self.plot_locations.split(',')]
                
            u_avg=self.Data['u_avg'].values
            TI=self.Data['u_stdev'].values/u_avg*100
            

            fig=plt.figure(figsize=(18,10))
                
            ctr=1

            for x_plot in x_plot_wake:
                u_avg_int=self.Data['u_avg'].interp(x=x_plot*D,method='linear').values
                u_std_int=self.Data['u_stdev'].interp(x=x_plot*D,method='linear').values
                TI_int=u_std_int/u_avg_int*100
                
                #Plot mean velocity
  
                ax = plt.subplot(len(x_plot_wake),2,(ctr-1)*2+1)
                if ctr==1:
                    plt.title('Mean streamwise velocity at '+self.inputData.attrs['location_id']+' on '+self.Data.attrs['start_time'][:8]+'\n File: '+os.path.basename(self.source)\
                              +'\n'+self.Data.attrs['start_time'][9:]+' - '+self.Data.attrs['end_time'][9:]+'\n'+r'$x='+str(x_plot)+'D$')
                else:
                    plt.title(r'$x='+str(x_plot)+'D$')
                        
                ax.set_facecolor((0,0,0,0.2))
                cf1=plt.contourf(y,z,u_avg_int.T,np.round(np.linspace(np.nanpercentile(u_avg,5), np.nanpercentile(u_avg,95), 10),2), cmap='coolwarm',extend='both')
                plt.grid(alpha=0.5)

                circle = Circle((0, 0), D/2, edgecolor='k', facecolor='none')
                ax.add_patch(circle)
                plt.plot(y,y*0+H,'k')
                
                plt.xlabel(r'$y$ [m]') 
                plt.ylabel(r'$z$ [m]')
                
                plt.xlim([self.ymin*D,self.ymax*D])
                plt.ylim([self.zmin*D,self.zmax*D])
                
                xlim=ax.get_xlim()
                ylim=ax.get_ylim()
                ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
                    


                #Plot TI
                
                    
                ax = plt.subplot(len(x_plot_wake),2,(ctr-1)*2+2)
                if ctr==1:
                    plt.title('Turbulence intensity at '+self.inputData.attrs['location_id']+' on '+self.Data.attrs['start_time'][:8]+'\n File: '+os.path.basename(self.source)\
                            +'\n'+self.Data.attrs['start_time'][9:]+' - '+self.Data.attrs['end_time'][9:]+'\n'+r'$x='+str(x_plot)+'D$')
                else:
                    plt.title(r'$x='+str(x_plot)+'D$')
                    
                ax.set_facecolor((0,0,0,0.2))
                cf2=plt.contourf(y,z,TI_int.T,np.unique(np.round(np.linspace(np.nanpercentile(TI,5), np.nanpercentile(TI,95), 20),1)), cmap='hot',extend='both')
                plt.grid(alpha=0.5)
                
                circle = Circle((0, 0), D/2, edgecolor='k', facecolor='none')
                ax.add_patch(circle)
                plt.plot(y,y*0+H,'k')
                
                plt.xlabel(r'$y$ [m]') 
                plt.ylabel(r'$z$ [m]')
               
                plt.xlim([self.ymin*D,self.ymax*D])
                plt.ylim([self.zmin*D,self.zmax*D])
                xlim=ax.get_xlim()
                ylim=ax.get_ylim()
                ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
                
                ctr+=1
            remove_labels(fig)
        
            axs=fig.axes
            cax = fig.add_axes([axs[-2].get_position().x0+axs[-2].get_position().width+0.0075,axs[-2].get_position().y0,
                                0.01,axs[0].get_position().y0+axs[0].get_position().height-axs[-2].get_position().y0])
            plt.colorbar(cf1,cax=cax,label=r'Mean streamwise velocity [m s$^{-1}$]')
            cax = fig.add_axes([axs[-1].get_position().x0+axs[-1].get_position().width+0.0075,axs[-1].get_position().y0,
                                0.01,axs[1].get_position().y0+axs[1].get_position().height-axs[-1].get_position().y0])
            plt.colorbar(cf2,cax=cax,label=r'Turbulence intensity [%]')
            
            fig.savefig(self.save_filename.replace('.nc','_cross-planes.png'))
            plt.close()
      
def mid(x):
    return (x[:-1]+x[1:])/2

def dt64_to_num(dt64):
    """
    numpy.datetime64[ns] time to Unix time
    """
    tnum=(dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return tnum

def datestr(num,format="%Y-%m-%d %H:%M:%S.%f"):
    """
    Unix time to string of custom format
    """
    from datetime import datetime
    
    string=datetime.utcfromtimestamp(num).strftime(format)
    return string
    
def LiSBOA_v7_2(x_exp,mins,maxs,Dn0,sigma,max_iter=None,calculate_stats=False,f=None,order=2,R_max=3,grid_factor=0.25,tol_dist=0.1,max_Dd=1,verbose=True):
    #03/01/2021 (v 5): undersampling checkes in hypercubes instead of hyperspheres (faster)
    #03/02/2022: finalized
    #08/28/2023 (v 7): added HOM,handled 0 dimensions, finalized
    #09/20/2023 (v 7.1): fixed bug on dimension number, handling grid in infinite dimension, finalized
    #03/05/2024 (v 7.2): spatial bins centered
    #03/06/2024: added verbosity, finalized
    
    from scipy.special import gamma
    from scipy.interpolate import interpn
    import itertools
    import sys
    import time
         
    n=3  
    
    #outliers rejection
    if calculate_stats:
        real=~np.isnan(np.sum(np.array(x_exp),axis=0)+f)
        for j in range(n):
            x_exp[j]=x_exp[j][real]
        f=f[real]
    else:
        real=~np.isnan(np.sum(np.array(x_exp),axis=0))
        for j in range(n):
            x_exp[j]=x_exp[j][real]
    
    #Initialization
    t0=time.time()
    Dn0=np.array(Dn0)+0.0 
    n_eff=np.sum(Dn0>0)
    Dn0[Dn0==0]=10**99
    N=len(x_exp[0])
    x=np.zeros((n,N))
    xc=np.zeros(n)
    X_bin=[];
    X_vec=[];
    X2=[]
    avg=None
    HOM=None
    
    #LiSBOA setup
    dx=grid_factor*Dn0
    R_max=R_max*sigma
    V=np.pi**(n_eff/2)/gamma(n_eff/2+1)*R_max**n_eff
    
    for j in range(n):
        xc[j]=np.nanmean(x_exp[j])
        x[j]=(x_exp[j]-xc[j])/Dn0[j]       
        X_bin.append((np.arange(mins[j]-dx[j]/2,maxs[j]+dx[j]/2+dx[j]*10**-10,dx[j])-xc[j])/Dn0[j])
        X_vec.append(mid(X_bin[j]))

    NoD=np.ceil(np.log10(np.max(np.abs(np.array(x)/tol_dist))))+1
    X=np.meshgrid(*[X_vec[j] for j in range(n)], indexing='ij')
    
    for j in range(n):
        X2.append(X[j]*Dn0[j]+xc[j])
      
    w=np.zeros(np.shape(X[0]),dtype=object)
    sel=np.zeros(np.shape(X[0]),dtype=object)
    val=np.zeros(np.shape(X[0]),dtype=object)
    Dd=np.zeros(np.shape(X[0]))
    N_grid=X[0].size
    dist_inf=np.zeros(n)
    
    #weights
    for j in range(n):
        dist_inf[j]=np.ceil(R_max/(dx[j]/Dn0[j]))
        
    nodes=np.where(X[0])
    counter=0
    for i in zip(*[xx for xx in nodes]):
        distSq=0
        for j in range(n):
            distSq+=(x[j]-X[j][i])**2
        s=np.where(distSq<R_max**2)   
        if len(s)>0:
            w[i]=np.exp(-distSq[s]/(2*sigma**2))
       
        #local spacing
        if Dd[i]!=10:   
            if len(s[0])>1:                
                pos_uni=np.around(x[0][s]/tol_dist)*tol_dist
                for j in range(1,n):                
                    pos_uni+=np.around(x[j][s]/tol_dist)*tol_dist*(10**NoD)**j          
                N_uni= len(np.unique(np.array(pos_uni)))
                
                if N_uni>1:
                    Dd[i]=V**(1/n_eff)/(N_uni**(1/n_eff)-1)
                else:
                    Dd[i]=np.inf
            else:
                Dd[i]=np.inf
                
            ind_inf=[]
            if Dd[i]>max_Dd:
                for j in range(n):
                    i1=max(i[j]-dist_inf[j],0)
                    i2=min(i[j]+dist_inf[j],np.shape(X[0])[j])
                    ind_inf.append(np.arange(i1,i2).astype(int))                
                for i_inf in itertools.product(*[ii for ii in ind_inf]):
                    Dd[i_inf]=10

        #store
        sel[i]=s
        
        counter+=1
        if np.floor(counter/N_grid*100)>np.floor((counter-1)/N_grid*100) and verbose:
            est_time=(time.time()-t0)/counter*(N_grid-counter)
            sys.stdout.write('\r LiSBOA:'+str(np.floor(counter/N_grid*100).astype(int))+'% done, '+str(round(est_time))+' s left.') 
    sys.stdout.write('\r                                                                         ')
    sys.stdout.flush()
    excl=Dd>max_Dd
                
    #stats
    if calculate_stats:
        avg=[]
        HOM=[]
        df=f
        for m in range(max_iter+1):
            WM=np.zeros(np.shape(X[0]))+np.nan
            WM_HOM=np.zeros(np.shape(X[0]))+np.nan
            if verbose:
                sys.stdout.write('\r Iteration #'+str(m))
                sys.stdout.flush()
            for i in zip(*[xx for xx in nodes]):
                val[i]=f[s]
                if not excl[i]:
                    fs=np.array(df[sel[i]])
                    ws=np.array(w[i])
                    reals=~np.isnan(fs+ws)
                    if sum(reals)>0:
                        fs=fs[reals]
                        ws=ws[reals]                     
                        WM[i]=sum(np.multiply(fs,ws))/sum(ws)              
                        if m>0:
                            WM_HOM[i]=sum(np.multiply(fs**order,ws))/sum(ws)  
            if m==0:
                avg.append(WM+0)
            else:
                avg.append(avg[m-1]+WM)
                HOM.append(WM_HOM)
            
            df=f-interpn(tuple(X_vec),avg[m],np.transpose(x),bounds_error=False,fill_value=np.nan)
        if verbose:
            sys.stdout.flush()
    return X2,Dd,excl,avg,HOM 

        
            
def remove_labels(fig):
    axs=fig.axes

    for ax in axs:
        try:
            loc=ax.get_subplotspec()
        
            if loc.is_last_row()==False:
                ax.set_xticklabels([])
                ax.set_xlabel('')
            if loc.is_first_col()==False:
                ax.set_yticklabels([])
                ax.set_ylabel('')
        except:
            pass

  
if __name__ == "__main__":

    cd=os.path.dirname(__file__)
        
    source='data/volumetric-wake-csm/rt1.lidar.z02.b0.20240304.032004.user5.awaken.wake.stats3d.nc'
    config_file='data/configs_statistics.xlsx'

    # Create an instance of LIDARGO
    lproc = LIDARGO(source,config_file, verbose=True)
    
    # Run processing
    lproc.process_scan(make_figures=True, replace=True,save_file=True)