from angels import angels as ang

#%% Inputs
m=14
n=3000
source_config="../configs/config.yaml"

#%% Main
noise_gen=ang.angels(source_config)

nn=noise_gen.generate_noise(m,n)




# def main():
#     # Generate a sample case of SNR and noise for a 180-degree RHI

#     # M and N come from the UTD Halo settings during AWAKEN
#     noise_snrs_db, noise_vel_stds = generate_noise_curve(16, 2985, u_nyq=39)

#     # SNR mean profiles for different clusters and standard deviation profile will be
#     # user provided (or can use the provided values extracted from the AWAKEN data)
#     snr_cluster_profiles_df = pd.read_csv("./rng_cor_snr_cluster_profiles.csv")
#     snr_std_profile_df = pd.read_csv("./std_height_curve.csv")

#     # Scan parameters
#     rng_gate = 48
#     max_rng = 4000
#     ele_vec = np.concatenate([
#         np.arange(0, 11, 1),
#         np.arange(11, 169+4, 4),
#         np.arange(170, 181, 1)
#     ])
#     ele_vec = np.deg2rad(ele_vec)
#     rng_vec = np.arange(rng_gate, max_rng+rng_gate, rng_gate)
#     ele_grid, rng_grid = np.meshgrid(ele_vec, rng_vec)
#     z_grid = rng_grid * np.sin(ele_grid)

#     # Generate a single instanteous sample of SNR
#     sampled_snr_db, fig0, ax0, fig1, ax1 = sample_snr(
#         rng_grid, z_grid,
#         snr_cluster_profiles_df["height"].to_numpy(),
#         snr_cluster_profiles_df["C2"].to_numpy(), 50,
#         snr_stds=snr_std_profile_df["snr*_std_db"].to_numpy(),
#         ele_grid=ele_grid
#     )

#     # From the sampled SNR and interpolated noise curve, sample RWS noise
#     sampled_noise = sample_noise(
#         sampled_snr_db, noise_snrs_db, noise_vel_stds, 39
#     )

#     # Plot the added noise
#     fig, ax = plt.subplots(1, 1, constrained_layout=True, subplot_kw={'projection':'polar'})
#     fig.set_size_inches([5, 5])

#     ax.set_xlim([0, np.pi])
#     sm = ax.pcolormesh(ele_grid, rng_grid, sampled_noise, cmap=VEL_CMAP,
#                        norm=mpl.colors.CenteredNorm(0, 10), rasterized=True)
#     ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
#     fig.colorbar(sm, label="Noise [m/s]", shrink=0.4)


# if __name__ == '__main__':
#     main()
#     plt.show()
