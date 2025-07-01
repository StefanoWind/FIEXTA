"""
General test for noise generator ANGELS. Read config file and README for more information.
"""
import angels as ang

#%% Inputs
m=14#FFT points or gate length
n=1000#accumulated pulses
cluster='C2'#cluster of SNR profile 
snr_0=50#[dB m^2] range-corrected SNR at the ground
n_samples=3#number of realizations of random noise
source_config="../configs/config.yaml"#path to config file

#%% Main
noise_gen=ang.angels(source_config)#load ANGLES class
output,fig_noise=noise_gen.generate_noise(m,n,cluster,snr_0,n_samples)#generate desired noise
