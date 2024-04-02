# LIDARGO
LIDAR General Operator: Standardization, quality control, and statistics of scanning lidar data

To test, run test.py on selected sample files. Sample files from AWAKEN and RAAW (NREL employees only) can be found at [sample data](https://nrel.app.box.com/folder/256760533953)

# LIDARGO_standardize
The standardization process identifyies nominal scan geometry and maps the beam angles on the regularized frame. The QC part flags suspect data points. 

All inputs are provided in the configuration file.

General inputs are:
- PARAMETER: it is the header used ot select the right configuration. It contains the regular expression needed to recognize the type of scan based on instrument location and time
- project: project name.
- name: humanly-readable identifier of the scan.
- data_level_in: string identifier of the input data level.
- data_level_out: string identifier of the output data level.
- azimuth_offset [deg]: offset of azimuth from true North (fixed lidar) or equivalent North for a turbine facing W wind. Azimuth is counted clowckwise.
- ground_level [m]: level of the ground with respect to the lidar head.

These parameters are used for the standardization:
-min_azi_step [deg]: minimum azimuth step, with sign.
-max_azi_step [deg]: maximum azimuth step, with sign.
-min_ele_step [deg]: minimum elevation step, with sign.
-max_ele_step [deg]: maximum elevation step, with sign.
-ang_tol [deg]: angular tolerance (keep it small).
-count_threshold [fraction of the maximum value]: minimum normalized number of occurrence of a certain azimuth/elevation pair.
-min_scan_duration [s]: minimum duration of a scan.

These are the inputs to the pre-filter:
-range_min [m]: minimum range.
-range_max [m]: maximum range.
-snr_min [dB]: minimum SNR.
-N_resonance_bins: number of bins used to calcuate the histogram of radial wind speed for the identification of the resonance of bad data around 0.
-max_resonance_rmse: maximum RMSE to consider of the histogram of bad radial wind speed data Gaussian, so affected by resonance.
-rws_max [m/s]: maximum absolute radial wind speed.

These are the inputs the dynamic filter:
-dx,dy,dz [m]: size of the bins in x (W-E or streamwise for nacelle-mounted lidars), y (S-N or lateral for nacelle-mounted lidars), and z (vertical) direction used in the dynamic filter to normalize radial wind speed and SNR.
-dtime [s]: size of the bins in time used in the dynamic filter to normalize radial wind speed and SNR.
-rws_norm_limit [m/s]: maximum absolute normalized radial wind speed.
-rws_standard_error_limit [m/s]: maximum standard error on the bin-median of radial wind speed used in the dynamic filter.
-snr_standard_error_limit [dB]: maximum standard error on the bin-median of SNR used in the dynamic filter.
-local_population_min_limit: minimum number of samples in each bin of the dynamic filter.
-local_scattering_min_limit: minimum ratio of good points within a bin of the dynamic filter.
-N_probability_bins: number of bins in the probability vs. normalized radial wind speed plot.
-min_percentile: lower percentile for the calculation of the dispersion of normalized radial wind speed within each probaility bin.
-max_percentile: upper percentile for the calculation of the dispersion of normalized radial wind speed within each probaility bin.
-rws_norm_increase_limit: maximum ratio of increase of dispersion of normalized radial wind speed over the full range to identify the probability threshold of the dynamic filter.

# LIDARGO_statistics (for nacelle-mounted lidars only)
Calculates time average and standard deviation of de-projected radial velocity data using the LiSBOA tool. It assumes wind coming from W, so it is suitable for nacelle-mounted lidars with negligible yaw offset.

All inputs are provided in the configuration file.

General inputs are:
-PARAMETER: it is the header used ot select the right configuration. It contains the regular expression needed to recognize the type of scan based on instrument location and time
-data_level_in: string identifier of the input data level.
-data_level_out: string identifier of the output data level.
-ground_level [m]: level of the ground with respect to the lidar head.
-diameter [m]: turbine diameter
-plot_locations [D]: where to slice y-z planes in the plot (only for 3-D scans).

These are the inputs to the LiSBOA:
-xmin,xmax [D]: minimum and maximum x (W-E or streamwise).
-ymin,ymax [D]: minimum and maximum y (W-E or lateral).
-ymin,ymax [D]: minimum and maximum z (vertical).
-Dn0_x,Dn0_y,Dn0_z [D]: fundamental half-wavelengths in x, y, z.
-sigma [fraction of fundamental half-wavelenghts]: smoothing paramter.
-max_iter: maximum number of iterations.



