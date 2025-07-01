import angels as ang

#%% Inputs
m=14
n=1000
cluster='C2'
snr_0=50
n_samples=10
source_config="../configs/config.yaml"

#%% Main
noise_gen=ang.angels(source_config)

sampled_noise, sampled_noise, x_grid, y_grid, z_grid=noise_gen.generate_noise(m,n,cluster,snr_0,n_samples)
