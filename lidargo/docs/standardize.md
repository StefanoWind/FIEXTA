## LIDARGO Standardize

The LIDARGO [standardize.py](https://github.com/StefanoWind/FIEXTA/blob/main/lidargo/standardize.py) is a class that converts previously formatted lidar data files into standardized and quality-controlled (QC) netCDF datasets. The process of standardization restructures the lidar data into an array with coordinates that facilitate the data analysis compared to the native coordinates. Formatted lidar wind speed and SNR data are stored as 2-D arrays with dimensions time and range gate index. This native structures is inconvenient for more advanced processing, so the standardization process recasts the data into a 3-D array with dimensions range, beam index, and scan repetition index. The QC process flags suspicious data points using a generalized outlier detection method based on the dynamic lidar filtering approach {cite}`Beck2017_RS_dlf`.

The general workflow of LIDARGO Standardize is outlined in {numref}`fig-standardize_workflow`. The detailed steps are described next.

```{figure} ./figures/standardize_workflow.png
---
name: fig-standardize_workflow
alt: fishy
class: bg-primary mb-1
width: 800px
align: center
---
Workflow of the LIDARGO Standardize class.
```

### Pre-conditioning
The geometry of raw lidar data is expressed in spherical coordinates as a function of range gate index (which is easily converted to physical distance, $r$), azimuth, $\alpha$, and elevation, $\beta$ ({numref}`fig-spherical`, left). Azimuth and elevation do not exactly match the nominal values provided by the lidar user due to imperfect pointing, which makes different scans inconsistent. On top of that, one data file is generally a collection of the data measured through the same scan patterns repeated several times with undesired data points in between that are acquired as the scanning head moves to home position at the beginning of each scan cycle.

```{figure} ./figures/spherical.png
---
name: fig-spherical
alt: fishy
class: bg-primary mb-1
width: 800px
align: center
---
Spherical and Cartesian coordinates as defined in LIDARGO.
```

The pre-conditioning process acts at the start of the standardization and goes through the following steps:
1. **Back-swipe removal**: data points acquired during unwanted movement of the scanning head are removed to keep only data collected through the nominal scan pattern. The off-design points to exclude are identified based on thresholds on the azimuth and elevation steps provided in the configuration file. Specifically, beams with azimuth outside the interval `[min_azi_step, max_azi_step]` or elevation outside the interval `[min_ele_step, max_ele_step]` are excluded.
2. **Nominal angles search**: since actual lidar angles meander around the prescribed scan geometry, nominal azimuth and elevation are identified as the most likely location within the give scan file. This is done by building a 2-D histogram of the lidar beam positions, $H(\alpha,\beta)$, in the azimuth and elevation space based on small bins of width `ang_tol`. Bins centers of bins with a normalized number of occurrences, that exceed the threshold `count_threshold` (i.e., $H(\alpha,\beta)/\text{max}(H(\alpha,\beta))\geq$`count_threshold`) are defined as the nominal angles. 
3. **Angles gridding**: after nominal angles are identified, the whole scan geometry is adjusted by slighlty shifting the raw azimuths and elevations to their closest nominal value. Beams falling more than `ang_tol` away from the closest nominal angle are flagged as off-design and discarded.

{numref}`fig-histogram_1d` shows the process of nominal angles search and gridding for a PPI scan run using the continuous scan mode on a Halo XR. The blue bars represent the occurrence of the raw azimuth angles which are clearly scattered around their nominal values. After the pre-conditioning, the angles fall in the more coarse grid indicated by the orange histogram, which simplifies the data analysis. 

```{figure} ./figures/rt1.lidar.z02.b0.20230830.063004.user5.awaken.meand.angHist.png
---
name: fig-histogram_1d
alt: fishy
class: bg-primary mb-1
width: 800px
align: center
---
Histogram of the azimuth for a PPI scan before and after the pre-processing.
```

{numref}`fig-ang_scatter` shows the time history of the angles before (blue) and after (orange) the pre-conditioning. This represent a successful case because most of the angle are retained and azimuths are just slighlty moved form their raw position by the gridding step. All these figures can be automatically generated by LIDARGO as a part of the `qc_report`. 

```{figure} ./figures/rt1.lidar.z02.b0.20230830.063004.user5.awaken.meand.angScatter.png
---
name: fig-ang_scatter
alt: fishy
class: bg-primary mb-1
width: 800px
align: center
---
Time history of the azimuth (top) and elevation (bottom) for a PPI scan before and after the pre-processing.
```
### Quality control
The QC in LIDARGO combines static and dynamic quality flags to identify outliers in the radial wind speed (RWS) data due to low-signal-to-noise ratio (SNR), which typically happen in the far range, echo from hard target, or signal saturation (first $\sim$ 100 m). The outcome is the `qc_wind_speed` structure which stores QC flags, where 0 indicates good data and other values bad data based on several criteria. The QC process includes the following steps:
1. **Dataframe conversion**: the xarray structure is converted into a pandas dataframe to facilitate the subsequent processing. At this stage LIDARGO also defines Cartesian coordinates for all points, as shown in {numref}`fig-spherical` and according to the follwing relationship:
```{math}
:label: eq-cartesian
\begin{cases}
x=r ~ \cos(\pi/2-(\alpha+\alpha_0)) ~ \cos(\beta)\\
y=r ~ \sin(\pi/2-(\alpha+\alpha_0)) ~ \cos(\beta)\\
z=r ~ \sin(\beta)\\
\end{cases}
```
where $\alpha_0$ is the `azimuth_offset`, which is added to the raw azimuth and can be used to align the Cartesian coordinate system to the desired direction (e.g., true north, turbine axis, etc.).

2. **Prefilter**: it applies static criteria to flag data that are easily identifiable as outliers. In particular, are flagged as bad data points with:
  1. Ranges outside the interval `[min_range,max_range]`.
  2. Heights below the ground level with respect to the lidar defined by `ground_level`.
  3. SNR lower than `snr_min`, which can be kept quite low (e.g., -25 dB) at this stage.
  4. RWS magnitude larger than `rws_max`, which can be kept quite high (e.g., 30 m/s) at this stage.
  5. RWS magnitude smaller than a threshold which is identified in the resonance detection step. Resonance refers to the clustering of outliers around 0 m/s that can happen in some lidars. The resonance detection analyzes the histogram of the RWS data flagged as bad to this point. If the histogram is not flat but can be fit with a Gaussian, the $2\sigma$ value of such fitting Gaussian function is taken as the minimum allowable value for the RWS magnitude to exclude the core of outliers close of 0 m/s. The user can control the number of bins in the histogram through `N_resonance_bins`, and the maximum Root Mean Square Error of the fit to consider the histogram a Guassian or `max_resonance_rmse`.
3. **Dynamic filter**: the concept of dynamic lidar filtering is introduced by {cite}`Beck2017_RS_dlf`, and it is based on the evidence that points with low probability in the 2-D histogram in the plane RWS-SNR are typically outliers. Before applying this criterion RWS and SNR are "normalized", meaning that the local median calculated over spatio-temporal bins is subtracted to isolate only the fluctuations on top of the mean field. The resulting histogram after the normalization is typically resemblant of a 2-D Gaussian, thus making this approach quite general regardless of the type of flow or scan geometry. The gist of the dynamic filter is illustrated in  {numref}`fig-dynamic_filter`. LIDARGO offers the users full control over the dynamic filter parameters, such as the bins sizes in space (`dx`, `dy`, `dz`) and time (`dtime`). Robustness of the bin-median is checked by imposing a minimum number of samples per bin (`local_population_min_limit`) and a maximum standard error on RWS and SNR (`rws_standard_error_limit`,`snr_standard_error_limit`). An intermediate static threshold on the maximum magnitude of the normalized RWS can be also imposed as `rws_norm_limit`.

:::{warning}
The use of the word "normalized" in the context of dynamic filter is a bit misleading. Normalized RWS and SNR are still dimensional quantities in m/s and dB, respectively, since just their local median has been removed but they are not actually divided by any reference value.
:::

```{figure} ./figures/dynamic_filter.png
---
name: fig-dynamic_filter
alt: fishy
class: bg-primary mb-1
width: 800px
align: center
---
How the dynamic filter works. The $\langle \rangle$ indicates the median operator. 
```
:::{note}
The choice of the size of the spatio-temproeral bins is the main and often only part of the dynamic filter that needs the user's input. These bins are supposed to remove the physical real variability present in the lidar data so a rule of thumbs is to choose their size close to the estimated spatial and temporal scales of the flow. For example, if you are scanning the wake of a turbine with diameter equal to 100 m and the $x$ axis is aligned with the rotor axis, a good choise is `dx=`200 m (roughly the length of the near wake region), `dy=dz=`100 m ($\sim$the wake width), and `dtime`=600 s (typical averaging time for atmospheric flows).
:::

The original dynamic filter does not provide a universal method to define the minimum probability thrshold to distinguis bad from good data. LIDARGO then adopts an innovative data-drive approach based on the evidence that good bins with high probability are also characterized by normalized RWS tightly confined around 0 m/s, while bad data with low probability are instead more dispersed around 0 m/s due to the high random noise (see  {numref}`fig-dynamic_filter_prob`, right). LIDARGO then identifies the probability threshold by calculating the normalized RWS range as a function of probability. The normalized RWS range is defined as a custom quantile range between `min_percentile` (e.g., 1%) and `max_percentile` (e.g., 99%), and calculated over `N_probability_bins` non-overlapping bins. The normalized RWS range as a function of probability generally shows a monotonic and often stepwise decrease with the probability iteslf ( {numref}`fig-dynamic_filter_prob`, right panel, orange line). The minimum probability threshold is then defined as the probability where the normalized RWS range, made non-dimensional by its maximum and minimum values, exceed a threshold named `rws_norm_increase_limit` ( {numref}`fig-dynamic_filter_prob`, right panel, green line).

:::{note}
Let's say that for a scan, the normalized RWS range averaged as a function probability goes from 1 m/s to 21 m/s and we impose `rws_norm_increase_limit=0.25`. Then the probability threshold is corresponds to the bin where the normalized RWS range exceeds:
1 + 0.25 (21-1) = 6 m/s
:::

This data-driven approach for the selection of the probability threshold works fairly well for a wide range of scans without need to change the default parameters. The probability theshold is however constrained in the interval `[min_probability_range, max_probability_range]` to enhance robustness. As final step of the QC, LIDARGO cleans up isolated good points by flagging as bad all data points belonging to spatio-temporal bins in $x$,$y$,$y$,$t$ that are populated by a fraction of bad points more than `local_scattering_min_limit`.

:::{warning}
How do you troubleshoot an unsatisfactory LIDARGO QC? First step is to check why a certain good (bad) data point has been wrongfully flagged as bad (good) by looking inside the `qc_wind_speed` flag. Then, the fix generally comes down to changing one of the configuration parameters, most likely the bin sizes `dx,dy,dz,dtime`. E.g., is a whole turbine wake in a large scan flagged as bad? Try decreasing `dx,dy,dz` to prevent physcial variability in the wake to be interpreted as noise in the normalized RWS.
:::

```{figure} ./figures/rt1.lidar.z02.b0.20230830.063004.user5.awaken.meand.probability.png
---
name: fig-dynamic_filter_prob
alt: fishy
class: bg-primary mb-1
width: 800px
align: center
---
Example of dynamic filter for a PPI scan.
```



```{bibliography}
:style: unsrt
```


