# LIDARGO standardize

The LIDARGO *standardize.py* is a class that converts previously formatted lidar data files into standardized and quality-controlled (QC) netDCF datasets. The process of standardization restructures the line-of-sight velocity data, $u_{LOS}$, into a 3-D array with coordinates that facilitate the data analysis compared to the native range and time coordinates. The QC process flags suspicious data points using a general outlier detection method based on the dynamic lidar filtering approach {cite}`Beck2017_RS_dlf`.

The general workflow of LIDARGO *standardize.py* is outlined in Figure {ref}`fig-standardize_workflow`.

```{figure} ./figures/standardize_workflow.png
---
name: fig-standardize_workflow
alt: fishy
class: bg-primary mb-1
width: 800px
align: center
---
Workflow of the LIDARGO *standardize.py* class.
```


```{bibliography}
:style: unsrt
```