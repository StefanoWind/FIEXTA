# LiDAR General Operation (LIDARGO)

LIDARGO is a general processor for raw lidar data that allows to reformat, standardize, quality-control and carry out objective statistics analysis.

The LIDARGO data nomenclature is inspired by the guidelines available at the [Wind Data Hub](https://a2e.energy.gov/login) and follows the following logic:

```{list-table} Description of data levels within LIDARGO.
:header-rows: 1
:name: tab-data_levels

* - Data level
  - Definition
  - Description
  - Format
  - Generator
* - 0x
  - Raw   
  - Raw data file as produced by the instrument
  - native
  - Lidar                       
* - ax    
  - Formatted  
  - Data reformated with minimal changes  
  - netCDF 
  - [format](format.md)  
* - bx    
  - Reviewed  
  - Data with QC flags  
  - netCDF 
  - [standardize](standardize.md) 
* - cx    
  - Derived  
  - Value-added product derived from bx level  
  - netCDF 
  - [statistics](statistics.md) 
```

The "x" is an index that is used to identify further layer within each data level. E.g., if b0 are data quality-controlled through a method X, b2 can be data quality-controlled through a method Y, generally more advanced.

In this context, each LIDARGO instance will generate output files of the data level specified in the table.



```{tableofcontents}
```

