# -*- coding: utf-8 -*-
"""
LiSBOA utilities
"""
from typing import Optional, Union
from lisboa.logger import SingletonLogger
import numpy as np
from functools import wraps
from lisboa.config import LisboaConfig
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

def get_logger(
        name: str = None, verbose: bool = True, logger: Optional[object] = None, filename=None
               ) -> SingletonLogger:
    """Utility function to get or create a logger instance."""
    
    #get logger only if it exists, otherwise create one
    if logger is None:
        logger = SingletonLogger(logger=logger, verbose=verbose,filename=filename)
    return logger


def with_logging(func):
    """Decorator to add logging to any class method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = self.logger

        if logger.verbose:
            class_name = self.__class__.__name__
            func_name = func.__name__
            logger.log(f"Calling {class_name}.{func_name}")

        try:
            result = func(self, *args, **kwargs)
            return result

        except Exception as e:
            if logger.verbose:
                class_name = self.__class__.__name__
                func_name = func.__name__
                logger.log(
                    f"Error in {class_name}.{func_name}: {str(e)}", level="error"
                )
            raise

    return wrapper

def _load_configuration(config: Union[dict, LisboaConfig]):
    """
    Load configuration from either dictionary, or LisboaConfig object.

    Args:
        config (dict, or LisboaConfig): Configuration source

    Returns:
        LisboaConfig or None: Configuration parameters or None if loading fails
    """
    try:
        if isinstance(config, LisboaConfig):
            return config, "Configuration successfully loaded"
        elif isinstance(config, dict):
            return LisboaConfig(**config), "Configuration successfully loaded"
        else:
            return None, f"Invalid config type. Expected dict or LisboaConfig, got {type(config)}"
            
    except Exception as e:
        return None, f"Error loading configuration: {str(e)}"


def mid(x):
    '''
    Midpoint of 1-D vector
    '''
    return (x[1:]+x[:-1])/2

def cosd(x):
    '''
    Cosine in degrees
    '''
    return np.cos(x/180*np.pi)

def sind(x):
    '''
    Sine in degrees
    '''
    return np.sin(x/180*np.pi)


def sphere2cart(r,azi,ele):
    x=np.outer(r,np.cos(np.radians(ele))*np.cos(np.radians(90-azi)))
    y=np.outer(r,np.cos(np.radians(ele))*np.sin(np.radians(90-azi)))
    z=np.outer(r,np.sin(np.radians(ele)))
    return x,y,z


def visualize_scan(Data):
    '''
    Visualize scan with well-sampled regions.
    '''

    #scan information
    azi=Data.azi.values
    ele=Data.ele.values
    r=Data.r.values
    info=r'$\alpha='+str(np.round(azi[0],2))+':'+str(np.round(np.diff(azi)[0],2))+':'+str(np.round(azi[-1],2))+r'^\circ$'+ '\n'+\
         r'$\beta=' +str(np.round(ele[0],2))+':'+str(np.round(np.diff(ele)[0],2))+':'+str(np.round(ele[-1],2))+r'^\circ$' + '\n'+\
         r'$r= '    +str(np.round(r[0],2))  +':'+str(np.round(np.diff(r)[0],2))  +':'+str(np.round(r[-1],2))+ '$ m' +'\n'+\
         r'$\Delta n_0='+str(Data.attrs['config_Dn0'])+r'$ m'                                         + '\n'+\
         r'$\epsilon_I='+str(np.round(Data.attrs['epsilon1'],2))+r'$'                      + '\n'+\
         r'$\epsilon_{II}='+str(np.round(Data.attrs['epsilon2'],2))+r'$'                   + '\n'+\
         r'$\tau_s='+str(np.round(Data.attrs['duration'],1))+r'$ s'
     
    coords=''
    for c in Data.excl.coords:
        coords+=c
        
    fig=plt.figure()
    
    #plot 2D scan
    if coords!='xyz':
        ax=plt.subplot(111)
        fill=np.zeros(np.shape(Data.excl))
        fill[Data.excl==False]=10
        plt.pcolor(Data[coords[0]],Data[coords[1]],fill.T,cmap='Greys',vmin=0,vmax=1,alpha=0.5)
        plt.plot(Data[coords[0]+'_points'],Data[coords[1]+'_points'],'.k',markersize=2)
        ax.set_aspect('equal')
        ax.set_xlim([Data.attrs['config_mins'][0],Data.attrs['config_maxs'][0]])
        ax.set_ylim([Data.attrs['config_mins'][1],Data.attrs['config_maxs'][1]])
        ax.set_xlabel(r'$'+str(coords[0])+'$ [m]')
        ax.set_ylabel(r'$'+str(coords[1])+'$ [m]')
        dtick=np.max([np.diff(ax.get_xticks())[0],
                      np.diff(ax.get_yticks())[0]])
        ax.set_xticks(np.arange(Data.attrs['config_mins'][0],Data.attrs['config_maxs'][0]+dtick,dtick))
        ax.set_yticks(np.arange(Data.attrs['config_mins'][1],Data.attrs['config_maxs'][1]+dtick,dtick))
        ax.text(0,Data.attrs['config_maxs'][1]/2,s=info,bbox={'edgecolor':'k','facecolor':(1,1,1,0.25)})

    #plot 3D scan
    elif coords=='xyz':
        ax=plt.subplot(111,projection='3d')
        x=Data[coords[0]].values
        y=Data[coords[1]].values
        z=Data[coords[2]].values
        x_exp=Data.x_points.values
        y_exp=Data.y_points.values
        z_exp=Data.z_points.values
        
        dx=np.diff(x)[0]
        dy=np.diff(y)[0]
        dz=np.diff(x)[0]
        
        fill=Data.excl.values==False
        X,Y,Z=np.meshgrid(np.append(x-dx/2,x[-1]+dx),np.append(y-dy/2,y[-1]+dy),np.append(z-dz/2,z[-1]+dz),indexing='ij')
        ax.voxels(X,Y,Z,fill,facecolor='g',alpha=0.1)
        
        sel_x=(x_exp>Data.attrs['config_mins'][0])*(x_exp<Data.attrs['config_maxs'][0])
        sel_y=(y_exp>Data.attrs['config_mins'][1])*(y_exp<Data.attrs['config_maxs'][1])
        sel_z=(z_exp>Data.attrs['config_mins'][2])*(z_exp<Data.attrs['config_maxs'][2])
        sel=sel_x*sel_y*sel_z
        plt.plot(x_exp[sel],y_exp[sel],z_exp[sel],'.k',markersize=2)
        ax.set_aspect('equal')
        ax.set_xlim([Data.attrs['config_mins'][0],Data.attrs['config_maxs'][0]])
        ax.set_ylim([Data.attrs['config_mins'][1],Data.attrs['config_maxs'][1]])
        ax.set_zlim([Data.attrs['config_mins'][2],Data.attrs['config_maxs'][2]])
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_zlabel(r'$z$ [m]')
        dtick=np.max([np.diff(ax.get_xticks())[0],
                      np.diff(ax.get_yticks())[0],
                      np.diff(ax.get_zticks())[0]])
        ax.set_xticks(np.arange(Data.attrs['config_mins'][0],Data.attrs['config_maxs'][0]+dtick,dtick))
        ax.set_yticks(np.arange(Data.attrs['config_mins'][1],Data.attrs['config_maxs'][1]+dtick,dtick))
        ax.set_zticks(np.arange(Data.attrs['config_mins'][2],Data.attrs['config_maxs'][2]+dtick,dtick))
        ax.text(0,0,Data.attrs['config_maxs'][2]/2,s=info,bbox={'edgecolor':'k','facecolor':(1,1,1,0.25)})
        
    return fig


def visualize_Pareto(Data):
    
    #extract information
    N_ang=len(Data.index_ang)
    N_dang=len(Data.index_dang)
    epsilon1=Data.epsilon1.values
    epsilon2=Data.epsilon2.values
    azi1=Data.azi1.values
    azi2=Data.azi2.values
    ele1=Data.ele1.values
    ele2=Data.ele2.values
    dazi=Data.dazi.values
    dele=Data.dele.values
    
    #plot Pareto front
    fig=plt.figure(figsize=(18,8))
    cmap = plt.cm.jet
    colors = [cmap(v) for v in np.linspace(0,1,len(Data.dazi))]
    N_row=int(np.floor(N_ang**0.5))
    N_col=int(np.ceil(N_ang/N_row))
    for i_ang in range(N_ang):
        plt.subplot(N_row,N_col,i_ang+1)
        plt.plot(epsilon1[np.arange(N_ang)!=i_ang,:],epsilon2[np.arange(N_ang)!=i_ang,:],'.k',markersize=30,alpha=0.25)
        for i_dang in range(N_dang):
            plt.plot(epsilon1[i_ang,i_dang],epsilon2[i_ang,i_dang],'.',
                     color=colors[i_dang],markeredgecolor='k',markersize=30,
                     label=r'$\Delta \alpha='+str(dazi[i_dang])+r'^\circ$, $\Delta \beta='+str(dele[i_dang])+r'^\circ$')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel(r'$\epsilon_I$')
        plt.ylabel(r'$\epsilon_{II}$')  
        plt.title(r'$\alpha \in ['+str(azi1[i_ang])+', '+str(azi2[i_ang])+r']^\circ$, $\beta \in ['+str(ele1[i_ang])+', '+str(ele2[i_ang])+r']^\circ$')
        plt.grid()
    plt.legend(draggable=True)
    
    return fig