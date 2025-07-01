# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:56:12 2025

@author: sletizia
"""

import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import truncnorm, uniform
from angels.utilities import get_logger, with_logging
from typing import Optional
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

class angels:
    def __init__(
        self,
        config: str,
        logger: Optional[object] = None,
        logfile=None,
    ):
        """
        Initialize the ANGELS.
    
        Args:
            config (str): Path to yaml config file
            logger (Logger, optional): Logger instance for logging messages. Defaults to None.
            logfile: Filename of existing logfile. Defaults to None.
        """
        
    
        self.logger = get_logger(logger=logger,filename=logfile)
        
        with open(config, "r") as file:
            self.config = yaml.safe_load(file)
            
        return
        
    @with_logging
    def generate_noise(self,m,n,cluster,snr_0,n_samples):
        
        #
        self.m=m
        self.n=n
        self.cluster=cluster
        self.snr_0=snr_0
        self.n_samples=n_samples
        
        #load noise std table
        noise_std=pd.read_excel(self.config["source_noise_std"],sheet_name='snr_1')
        self.snr_1=pd.read_excel(self.config["source_noise_std"],sheet_name='snr_1').values[:,1:]
        self.sigma_t1=pd.read_excel(self.config["source_noise_std"],sheet_name='sigma_t1').values[:,1:]
        self.m_vec=np.array([int(m) for m in noise_std["M/N"].values])
        self.n_vec=np.array([int(n) for n in noise_std.columns[1:]])
        
        #noise st.dev vs. SNR
        noise_snr_db, noise_vel_std = self.generate_noise_curve(m, n, u_nyq=self.config["u_nyquist"])
        
        #SNR cluster stats
        snr_height, snr_corr_norm_avg, snr_corr_std=self.read_snr_stats(cluster)
        
        #sampled SNR
        sampled_snr_db,x_grid,y_grid,z_grid,rng_vec= self.sample_snr(snr_height, snr_corr_norm_avg, snr_corr_std, snr_0,n_samples)
        
        #generate noise
        sampled_noise = self.sample_noise(sampled_snr_db, noise_snr_db, noise_vel_std, self.config["u_nyquist"])
        
        fig_noise=self.visualizations(sampled_noise, sampled_snr_db, x_grid, y_grid, z_grid)
        
        #output
        output=xr.Dataset()
        output['rws_noise']=xr.DataArray(data=sampled_noise,coords={'range':rng_vec,'beamID':np.arange(len(self.azi)),'realization':np.arange(n_samples)},
                                        attrs={'units':'m/s','description':'noise of radial wind speed'})
        output['snr']=xr.DataArray(data=sampled_snr_db,coords={'range':rng_vec,'beamID':np.arange(len(self.azi)),'realization':np.arange(n_samples)},
                                        attrs={'units':'dB','description':'signal-to-noise ratio'})
        output.attrs["M"]=m
        output.attrs["N"]=n
        output.attrs["cluster"]=cluster
        output.attrs["SNR_0"]=snr_0
        for c in self.config:
            output.attrs["config_"+c]=self.config[c]
        output.attrs["contact"]='Coleman Moss (coleman.moss@utdallas.edu)'
        output.attrs["description"]="Lidar noise simulated based on Moss, Letizia, Iungo 2025"
    
        return output, fig_noise

    def generate_noise_curve(self, m, n, u_nyq, fig=False):
        '''
        Take gate length points 'm' and number of pulses 'n' and return an array of SNR values
        and associated noise standard deviations using Eq. 7 from the paper. SNR_1 and 
        sigma_{T,1} are interpolated from m and n using the table of points given in Fig. 15.
        '''
        if (m > np.max(self.m_vec)) or (m < np.min(self.m_vec)):
            self.logger.log("M is out of bounds")
            
        if (n > np.max(self.n_vec)) or (n < np.min(self.n_vec)):
            self.logger.log("N is out of bounds")
    
        #define low SNR regime
        snr2 = self.config["snr_2"]
        sigma_t2 = 2 / np.sqrt(12) * u_nyq

        #define SNR grid
        snr_lims = self.config["snr_lim"]
        snr_points = np.linspace(snr_lims[0], snr_lims[1], self.config["n_snr_points"])
        log_std = np.full((snr_points.shape[0], ), np.nan)
    
        #extract from look-up table
        n_grid,m_grid=np.meshgrid(self.n_vec,self.m_vec)
        snr_interper = LinearNDInterpolator(np.concatenate([
            n_grid.reshape(-1, 1),
            m_grid.reshape(-1, 1)
        ], axis=-1), self.snr_1.reshape(-1, 1))
        snr1 = snr_interper(np.array([n, m]))[0, 0]
    
        sigma_interper = LinearNDInterpolator(np.concatenate([
            n_grid.reshape(-1, 1),
            m_grid.reshape(-1, 1)
        ], axis=-1), self.sigma_t1.reshape(-1, 1))
        sigma_t1 = sigma_interper(np.array([n, m]))[0, 0]
    
        region1=snr_points <= snr2
        region2=(snr2 < snr_points) & (snr_points <= snr1)
        region3=snr_points > snr1
        log_std[region1] = np.log(sigma_t2)
        log_std[region2] = np.log(sigma_t2) - (np.log(sigma_t2) - np.log(sigma_t1))*((snr_points[region2] - snr2)/(snr1-snr2))**10
        log_std[region3] = np.log(sigma_t1) - np.log(10)/10*(snr_points[region3] - snr1)
        noise_std = np.exp(log_std)
    
        if fig:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            fig.set_size_inches([3, 3])
    
            plot_lims = [-30, -10]
            plot_mask = (snr_points >= plot_lims[0]) & (snr_points <= plot_lims[1])
            ax.semilogy(snr_points[plot_mask], noise_std[plot_mask], 'k-', lw=2)
            ax.grid(True)
            ax.set_xlabel("SNR [dB]")
            ax.set_ylabel("$\\sigma_T$ [m s$^{-1}$]")
    
        self.logger.log(f"Fitted sigma_T1: {sigma_t1:.3f}")
        self.logger.log(f"Fitted SNR_1: {snr1:.3f}")
    
        return snr_points, noise_std
    
    def read_snr_stats(self,cluster):
        """
        Read mean of range-corrected, normalized SNR and st. dev. of range-corrected SNR from cluster analysis.
        """
        
        try:
            snr_height = pd.read_excel(self.config["source_snr_stats"])["height"].values
        except:
            self.logger.log(f'Could not return height from {self.config["source_snr_stats"]}.')
            return None
        
        try:
            snr_corr_norm_avg = pd.read_excel(self.config["source_snr_stats"])[cluster].values
        except:
            self.logger.log(f'Could not return mean normalized range-corrected SNR from {self.config["source_snr_stats"]} for cluster {cluster}.')
            return None
        try:
            snr_corr_std = pd.read_excel(self.config["source_snr_stats"])["snr_std"].values
        except:
            self.logger.log(f'Could not return st.dev of range-corrected SNR from {self.config["source_snr_stats"]}.')
            return None
        
        return snr_height, snr_corr_norm_avg, snr_corr_std
    
    def read_scan_geometry(self):
        
        try:
            azi_vec = pd.read_excel(self.config["source_scan_geometry"])["Azimuth"].values
            ele_vec = pd.read_excel(self.config["source_scan_geometry"])["Elevation"].values
        except:
            self.logger.log(f'Could not load scan geometry from {self.config["source_snr_stats"]}.')
            return None
        
        self.azi=azi_vec
        self.ele=ele_vec
        return azi_vec,ele_vec
        
    def sample_snr(self, snr_height,snr_corr_norm_avg,snr_corr_std, snr_0, n_samples):
        '''
        For a given grid of range and height points, a curve of range-corrected snr with
        associated heights, the normalization parameter snr_0, and the standard deviation of
        SNR (in dB) at the same heights as the range-corrected profile, generate a single sample
        of SNR.
        '''
        
        #read and expand scan geometry
        azi_vec,ele_vec=self.read_scan_geometry()
        rng_vec = np.arange(self.config["rng_gate"], self.config["max_rng"]+self.config["rng_gate"], self.config["rng_gate"])
        rng_grid=np.outer(rng_vec,np.zeros(len(ele_vec))+1)
        x_grid=np.outer(rng_vec,np.cos(np.radians(90-azi_vec))*np.cos(np.radians(ele_vec)))
        y_grid=np.outer(rng_vec,np.sin(np.radians(90-azi_vec))*np.cos(np.radians(ele_vec)))
        z_grid=np.outer(rng_vec,np.sin(np.radians(ele_vec)))
        
        #full map of range-corrected mean SNR
        snr_corr_norm_avg_grid_db = np.interp(z_grid.flatten(), snr_height, snr_corr_norm_avg).reshape(z_grid.shape)
        snr_corr_avg_grid = 10**(snr_corr_norm_avg_grid_db * snr_0 / 10)
        
        #full map of mean SNR
        snr_avg_grid = np.full(snr_corr_avg_grid.shape, np.nan)
        
        #far range/high altitude
        region1=(rng_grid > self.config["rng_min"]) & (z_grid > self.config["z_min"])
        snr_avg_grid[region1] = snr_corr_avg_grid[region1] / rng_grid[region1]**2
        
        #short range/high altitude
        region2=(rng_grid <= self.config["rng_min"]) & (z_grid > self.config["z_min"])
        snr_avg_grid[region2] = snr_corr_avg_grid[region2] / self.config["rng_min"]**2
        
        #far range/short altitude
        region3=(rng_grid > self.config["rng_min"]) & (z_grid <= self.config["z_min"])
        snr_avg_grid[region3] = 10**(snr_0 / 10)/rng_grid[region3]**2
        
        #short range/short altitude
        region4=(rng_grid <= self.config["rng_min"]) & (z_grid <= self.config["z_min"])
        snr_avg_grid[region4] = 10**(snr_0 / 10)/self.config["rng_min"]**2
        
        snr_avg_grid[snr_avg_grid < self.config["snr_min"]] = self.config["snr_min"]
        # snr_avg_grid_db = 10*np.log10(snr_avg_grid)
        snr_corr_avg_grid = snr_avg_grid * rng_grid**2
    
        # Assume that snr std is constant below the minimum height
        # np.interp fills the values appropriately when beyond the limits
        snr_corr_std_grid_db = np.interp(z_grid.flatten(), snr_height, snr_corr_std).reshape(z_grid.shape)
        snr_corr_std_grid = 10**(snr_corr_std_grid_db/10)
    
        sampled_snr_corr = np.random.normal(snr_corr_avg_grid[..., np.newaxis],
                                            snr_corr_std_grid[..., np.newaxis],
                                            size=(np.shape(rng_grid)[0], np.shape(rng_grid)[1], n_samples))
        sampled_snr = sampled_snr_corr / np.repeat(rng_grid[..., np.newaxis],n_samples,axis=2)**2
        sampled_snr[sampled_snr < self.config["snr_min"]] = self.config["snr_min"]
        sampled_snr_db = 10*np.log10(sampled_snr)
    
        return sampled_snr_db,x_grid,y_grid,z_grid,rng_vec

    def sample_noise(self,snr_grid_db, noise_snr_db, noise_vel_std,u_nyq):
        '''
        For a 2D grid of SNR points (in dB) representing the measurement field of SNR of a
        single sample, generate a sample of velocity noise. The 'noise_snrs_db' is an array of
        SNR values at which to interpolate 'noise_vel_std' and get the standard deviation of
        the added noise.
        '''
        uni_std = 2 * u_nyq / np.sqrt(12)
        noise_vel_std_grid = np.interp(snr_grid_db.flatten(), noise_snr_db, noise_vel_std).reshape(snr_grid_db.shape)
        normal_weight = 5/4 * (1 - (noise_vel_std_grid/uni_std)**2)
        uniform_weight = 1 - normal_weight

        normal_weight[normal_weight < 0] = 0
        normal_weight[normal_weight > 1] = 1
        uniform_weight[uniform_weight < 0] = 0
        uniform_weight[uniform_weight > 1] = 1

        uniform_std = np.full(snr_grid_db.shape, uni_std)
        normal_std = np.sqrt((noise_vel_std_grid**2 - uniform_std**2 * uniform_weight) / normal_weight)
        normal_std[np.isnan(normal_std)] = 0
        normal_std[np.isinf(normal_std)] = 0

        sampled_noise = np.full(normal_std.shape, np.nan)
        for k in range(normal_std.shape[2]):
            for i in range(normal_std.shape[0]):
                for j in range(normal_std.shape[1]):
                    sampled_noise[i, j, k] = self.sample_from_joint_distribution(
                        normal_std[i, j, k], normal_weight[i, j, k], u_nyq
                    )
            self.logger.log(f'Noise generation: {k+1}/{normal_std.shape[2]} scans done.')
        return sampled_noise
    
    def sample_from_joint_distribution(self,normal_std, normal_weight, u_nyq):
        if normal_weight == 0:
            return np.random.uniform(-u_nyq, u_nyq)
        x_points = np.linspace(-u_nyq, u_nyq, 100)
        normal_cdf = truncnorm.cdf(x=x_points, a=-u_nyq, b=u_nyq, loc=0, scale=normal_std)
        normal_cdf *= normal_weight
        uni_cdf = uniform.cdf(x_points, -u_nyq, u_nyq*2)
        uni_cdf *= (1 - normal_weight)
        joint_cdf = normal_cdf + uni_cdf

        # Sample from joint distribution
        uni_samp = np.random.uniform(0, 1)
        joint_samp = np.interp(uni_samp, joint_cdf, x_points)

        return joint_samp
    
    def identify_scan_mode(self,azi,ele):
        """
        Identify the type of scan, which is useful for plotting
        """
        azimuth_variation = np.abs(np.nanmax(np.tan(azi)) - np.nanmin(np.tan(azi)))
        elevation_variation = np.abs(np.nanmax(np.cos(ele)) - np.nanmin(np.cos(ele)))

        if (elevation_variation < 0.25
            and azimuth_variation < 0.25
        ):
            scan_mode= "Stare"
        elif (
            elevation_variation < 0.25
            and azimuth_variation > 0.25
        ):
            scan_mode = "PPI"
        elif (
            elevation_variation > 0.25
            and azimuth_variation < 0.25
        ):
            scan_mode = "RHI"
        else:
            scan_mode = "3D"
        return scan_mode
    
    def visualizations(self,sampled_noise, sampled_snr_db, x_grid, y_grid, z_grid):
        """
        Visualization of SNR and noise
        """
        
        scan_mode=self.identify_scan_mode(self.azi,self.ele)
        
        if scan_mode=='stare':
            fig_noise = plt.figure(figsize=(12, 12))
            ax=plt.subplot(2,1,1)
            rng_vec = np.arange(self.config["rng_gate"], self.config["max_rng"]+self.config["rng_gate"], self.config["rng_gate"])
            plt.pcolor(np.arange(self.n_samples),rng_vec,sampled_noise.squeeze(),vmin=-self.config["u_nyquist"],vmax=self.config["u_nyquist"],cmap='seismic')

            plt.xlabel("Realization")
            plt.ylabel("Range [m]")
            plt.grid()
            plt.title(f"M={self.m}, N={self.n}, cluster={self.cluster},"+r" SNR$_0$="+f"{self.snr_0} dB")
            plt.colorbar(label="Radial wind speed noise [m s$^{-1}$]")
            
            ax=plt.subplot(2,1,2)
            plt.pcolor(np.arange(self.n_samples),rng_vec,sampled_snr_db.squeeze(),vmin=-30,vmax=0,cmap='hot')

            plt.xlabel("Realization")
            plt.ylabel("Range [m]")
            plt.grid()
            plt.colorbar(label="SNR [dB]")
            plt.tight_layout()
            
        elif scan_mode=='RHI':
            fig_noise = plt.figure(figsize=(12, 12))
            ax=plt.subplot(2,1,1)
            plt.pcolor((x_grid**2+y_grid**2)**0.5*np.sign(np.cos(np.radians(self.ele))),z_grid,sampled_noise[:,:,0],vmin=-self.config["u_nyquist"],vmax=self.config["u_nyquist"],cmap='seismic')

            ax.set_aspect("equal")
            plt.xlabel("Horizontal dimension [m]")
            plt.ylabel("Vertical dimension [m]")
            plt.grid()
            plt.title(f"M={self.m}, N={self.n}, cluster={self.cluster},"+r" SNR$_0$="+f"{self.snr_0} dB")
            plt.colorbar(label="Radial wind speed noise [m s$^{-1}$]")
            
            ax=plt.subplot(2,1,2)
            plt.pcolor((x_grid**2+y_grid**2)**0.5*np.sign(np.cos(np.radians(self.ele))),z_grid,sampled_snr_db[:,:,0],vmin=-30,vmax=0,cmap='hot')

            ax.set_aspect("equal")
            plt.xlabel("Horizontal dimension [m]")
            plt.ylabel("Vertical dimension [m]")
            plt.grid()
            plt.colorbar(label="SNR [dB]")
            plt.tight_layout()
            
        elif scan_mode=='PPI':
            fig_noise = plt.figure(figsize=(12, 12))
            ax=plt.subplot(2,1,1)
            plt.pcolor(x_grid,y_grid,sampled_noise[:,:,0],vmin=-self.config["u_nyquist"],vmax=self.config["u_nyquist"],cmap='seismic')
            ax.set_aspect("equal")
            plt.xlabel("X dimension [m]")
            plt.ylabel("Y dimension [m]")
            plt.grid()
            plt.title(f"M={self.m}, N={self.n}, cluster={self.cluster},"+r" SNR$_0$="+f"{self.snr_0} dB")
            plt.colorbar(label="Radial wind speed noise [m s$^{-1}$]")
            
            ax=plt.subplot(2,1,2)
            plt.pcolor(x_grid,y_grid,sampled_snr_db[:,:,0],vmin=-30,vmax=0,cmap='hot')
            ax.set_aspect("equal")
            plt.xlabel("X dimension [m]")
            plt.ylabel("Y dimension [m]")
            plt.grid()
            plt.colorbar(label="SNR [dB]")
            plt.tight_layout()
            
        elif scan_mode=='3D':
            fig_noise = plt.figure(figsize=(12, 12))
            ax = plt.subplot(2,1,1, projection="3d")
            sc = ax.scatter(x_grid,y_grid,z_grid,s=2,c=sampled_noise[:,:,0],
                 vmin=-self.config["u_nyquist"],vmax=self.config["u_nyquist"],cmap='seismic')

            ax.set_aspect("equal")
            ax.set_xlabel("X dimension [m]")
            ax.set_ylabel("Y dimension [m]")
            ax.set_zlabel("Vertical dimension [m]")
            plt.grid()
            plt.title(f"M={self.m}, N={self.n}, cluster={self.cluster},"+r" SNR$_0$="+f"{self.snr_0} dB")
            plt.colorbar(sc,label="Radial wind speed noise [m s$^{-1}$]")
            
            ax = plt.subplot(2,1,2, projection="3d")
            sc = ax.scatter(x_grid,y_grid,z_grid,s=2,c=sampled_snr_db[:,:,0],
                 vmin=-30,vmax=0,cmap='hot')

            ax.set_aspect("equal")
            ax.set_xlabel("X dimension [m]")
            ax.set_ylabel("Y dimension [m]")
            ax.set_zlabel("Vertical dimension [m]")
            plt.grid()
            plt.title(f"M={self.m}, N={self.n}, cluster={self.cluster},"+r" SNR$_0$="+f"{self.snr_0} dB")
            plt.colorbar(sc,label="SNR [dB]")
            plt.tight_layout()
            
        return fig_noise