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
import matplotlib as mpl
import matplotlib.pylab as pl
from scipy.stats import truncnorm, uniform
import cmasher as cmr
from lidargo.utilities import get_logger, with_logging, _load_configuration
from typing import Optional


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
        
        self.config=yaml.read(config)
    
    @with_logging
    def generate_noise(self,m,n,config):
        
        #load noise std table
        noise_std_table=pd.read_csv(self.config["source_noise_std_table"])
        self.m_vec=noise_std_table["M/N"]
        self.n_vec=noise_std_table.columns[1:]
        noise_snrs_db, noise_vel_stds = self.generate_noise_curve(m, n, u_nyq=self.config["u_nyquist"])


    def generate_noise_curve(self, m, n, u_nyq, snr_lims=None, fig=False):
        '''
        Take gate length points 'm' and number of pulses 'n' and return an array of SNR values
        and associated noise standard deviations using Eq. 7 from the paper. SNR_1 and 
        sigma_{T,1} are interpolated from m and n using the table of points given in Fig. 15.
        '''
        if (m <= np.max(self.m_vec)) or (m >= np.min(self.m_vec)):
            self.logger.log("M is out of bounds")
            
        if (n <= np.max(self.n_vec)) or (n >= np.min(self.n_vec)):
            self.logger.log("N is out of bounds")
    
        snr2 = -30
        sigma_t2 = 2 / np.sqrt(12) * u_nyq
        if snr_lims is None:
            snr_lims = [-50, 10]
        snr_points = np.linspace(snr_lims[0], snr_lims[1], 200)
        log_stds = np.full((snr_points.shape[0], ), np.nan)
    
        snr_interper = LinearNDInterpolator(np.concatenate([
            self.n_grid.reshape(-1, 1),
            self.m_grid.reshape(-1, 1)
        ], axis=-1), self.snr_1.reshape(-1, 1))
        snr1 = snr_interper(np.array([n, m]))[0, 0]
    
        sigma_interper = LinearNDInterpolator(np.concatenate([
            self.n_grid.reshape(-1, 1),
            self.m_grid.reshape(-1, 1)
        ], axis=-1), self.sigma_t1.reshape(-1, 1))
        sigma_t1 = sigma_interper(np.array([n, m]))[0, 0]
    
        log_stds[snr_points <= snr2] = np.log(sigma_t2)
        log_stds[(snr2 < snr_points) & (snr_points <= snr1)] = np.log(sigma_t2) - (np.log(sigma_t2) - np.log(sigma_t1)) * np.power((snr_points[(snr2 < snr_points) & (snr_points <= snr1)] - snr2)/(snr1-snr2), 10)
        log_stds[snr_points > snr1] = np.log(sigma_t1) - np.log(10)/10*(snr_points[snr_points > snr1] - snr1)
        noise_stds = np.exp(log_stds)
    
        if fig:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            fig.set_size_inches([3, 3])
    
            plot_lims = [-30, -10]
            plot_mask = (snr_points >= plot_lims[0]) & (snr_points <= plot_lims[1])
            ax.semilogy(snr_points[plot_mask], noise_stds[plot_mask], 'k-', lw=2)
            ax.grid(True)
            ax.set_xlabel("SNR [dB]")
            ax.set_ylabel("$\\sigma_T$ [m s$^{-1}$]")
    
        print(f"Fitted sigma_T1: {sigma_t1:.3f}")
        print(f"Fitted SNR_1: {snr1:.3f}")
    
        return snr_points, noise_stds


# '''
# For a given grid of range and height points, a curve of range-corrected snr with
# associated heights, the normalization parameter snr_0, and the standard deviation of
# SNR (in dB) at the same heights as the range-corrected profile, generate a single sample
# of SNR. The minimum range, height, and SNR are all hard-coded according to the paper. If
# grid of elevation points is also provided, figures are generated of the mean and sampled
# SNR fields and the figures and axes are returned along with the SNR field. Otherwise,
# just the SNR field is returned.
# '''
# def sample_snr(rng_grid, z_grid, heights, snrs, snr_0, snr_stds, ele_grid=None):
#     rng_min = 500
#     z_min = 50
#     min_snr = 0.0016
#     mean_snr_cor = np.interp(z_grid.flatten(), heights, snrs).reshape(z_grid.shape)
#     mean_snr_cor = np.power(10, mean_snr_cor * snr_0 / 10)
#     mean_snr = np.full(mean_snr_cor.shape, np.nan)
#     mean_snr[(rng_grid > rng_min) & (z_grid > z_min)] = mean_snr_cor[(rng_grid > rng_min) & (z_grid > z_min)] / rng_grid[(rng_grid > rng_min) & (z_grid > z_min)]**2
#     mean_snr[(rng_grid <= rng_min) & (z_grid > z_min)] = mean_snr_cor[(rng_grid <= rng_min) & (z_grid > z_min)] / rng_min**2
#     mean_snr[(rng_grid > rng_min) & (z_grid <= z_min)] = np.power(10, snr_0 / 10) / rng_grid[(rng_grid > rng_min) & (z_grid <= z_min)]**2
#     mean_snr[(rng_grid <= rng_min) & (z_grid <= z_min)] = np.power(10, snr_0 / 10) / rng_min**2
#     mean_snr[mean_snr < min_snr] = min_snr
#     mean_snr_db = 10*np.log10(mean_snr)
#     mean_snr_cor = mean_snr * rng_grid**2

#     # Assume that snr std is constant below the minimum height
#     # np.interp fills the values appropriately when beyond the limits
#     snr_std_cor = np.interp(z_grid.flatten(), heights, snr_stds).reshape(z_grid.shape)
#     snr_std_cor = np.power(10, snr_std_cor/10)

#     sampled_snr_cor = np.random.normal(mean_snr_cor, snr_std_cor)
#     sampled_snr = sampled_snr_cor / rng_grid**2
#     sampled_snr[sampled_snr < min_snr] = min_snr
#     sampled_snr_db = 10*np.log10(sampled_snr)

#     if ele_grid is not None:
#         fig0, ax0 = plt.subplots(1, 1, constrained_layout=True, subplot_kw={'projection':'polar'})
#         fig0.set_size_inches([5, 5])

#         vmin = -28
#         vmax = 0

#         ax0.set_xlim([0, np.pi])
#         sm = ax0.pcolormesh(ele_grid, rng_grid, mean_snr_db, cmap=SNR_CMAP, vmin=vmin, vmax=vmax, rasterized=True)
#         ax0.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
#         fig0.colorbar(sm, label="SNR [dB]", shrink=0.4)

#         fig1, ax1 = plt.subplots(1, 1, constrained_layout=True, subplot_kw={'projection':'polar'})
#         fig1.set_size_inches([5, 5])

#         ax1.set_xlim([0, np.pi])
#         sm = ax1.pcolormesh(ele_grid, rng_grid, sampled_snr_db, cmap=SNR_CMAP, vmin=vmin, vmax=vmax, rasterized=True)
#         ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
#         fig1.colorbar(sm, label="SNR [dB]", shrink=0.4)
#         return sampled_snr_db, fig0, ax0, fig1, ax1

#     return sampled_snr_db


# '''
# Sample from the joint uniform and normal distribution described by the normal standard
# deviation, weight of the normal component, and u_nyq, which gives the standard deviation
# of the uniform distribution. 'n_samps' are taken and return.
# '''
# def sample_from_joint_distribution(norm_std, norm_weight, n_samps, u_nyq):
#     if norm_weight == 0:
#         return np.random.uniform(-u_nyq, u_nyq, n_samps)
#     x_points = np.linspace(-u_nyq, u_nyq, 100)
#     norm_cdf = truncnorm.cdf(x=x_points, a=-u_nyq, b=u_nyq, loc=0, scale=norm_std)
#     norm_cdf *= norm_weight
#     uni_cdf = uniform.cdf(x_points, -u_nyq, u_nyq*2)
#     uni_cdf *= (1 - norm_weight)
#     joint_cdf = norm_cdf + uni_cdf

#     # Sample from joint distribution
#     uni_samp = np.random.uniform(0, 1, n_samps)
#     joint_samp = np.interp(uni_samp, joint_cdf, x_points)

#     return joint_samp


# '''
# For a 2D grid of SNR points (in dB) representing the measurement field of SNR of a
# single sample, generate a sample of velocity noise. The 'noise_snrs_db' is an array of
# SNR values at which to interpolate 'noise_vel_stds' and get the standard deviation of
# the added noise.
# '''
# def sample_noise(snr_grid_db, noise_snrs_db, noise_vel_stds, u_nyq):
#     uni_std = 2 * u_nyq / np.sqrt(12)
#     noise_vel_std_grid = np.interp(snr_grid_db.flatten(), noise_snrs_db, noise_vel_stds).reshape(snr_grid_db.shape)
#     normal_weights = 5 / 4 * (1 - (noise_vel_std_grid/uni_std)**2)
#     uniform_weights = 1 - normal_weights

#     normal_weights[normal_weights < 0] = 0
#     normal_weights[normal_weights > 1] = 1
#     uniform_weights[uniform_weights < 0] = 0
#     uniform_weights[uniform_weights > 1] = 1

#     uniform_stds = np.full(snr_grid_db.shape, uni_std)
#     desired_stds = np.interp(snr_grid_db.flatten(), noise_snrs_db, noise_vel_stds).reshape(snr_grid_db.shape)
#     normal_stds = np.sqrt((desired_stds**2 - uniform_stds**2 * uniform_weights) / normal_weights)
#     normal_stds[np.isnan(normal_stds)] = 0
#     normal_stds[np.isinf(normal_stds)] = 0

#     sampled_noise = np.full(normal_stds.shape, np.nan)
#     for i in range(normal_stds.shape[0]):
#         for j in range(normal_stds.shape[1]):
#             sampled_noise[i, j] = sample_from_joint_distribution(
#                 normal_stds[i, j], normal_weights[i, j], 1, u_nyq
#             )
#     return sampled_noise

