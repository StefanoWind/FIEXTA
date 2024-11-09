# LIDARGO_format

*LIDARGO_format.py* is a class that converts raw lidar data files into netCDF datasets compatible with *LIDARGO_standardize.py* and compliant with the [Wind Data Hub](https://a2e.energy.gov/login) standards.

The class initialization requires the folling inputs:
```{list-table} Inputs to LIDARGO_format initialization.
:header-rows: 1
:name: tab-LiDARGO_format1

* - Input
  - Format
  - Units
  - Description
* - verbose
  - boolean   
  - N/A
  - Whether to print debug information
* - logger
  - logger object   
  - N/A
  - External debug and error log

```

The main module is the *process_scan* which accepts the following inputs:

```{list-table} Inputs to *process_scan* in LIDARGO_format.
:header-rows: 1
:name: tab-LiDARGO_format2

* - Input
  - Format
  - Units
  - Description
* - source
  - str   
  - N/A
  - Path to the 0x-level file to format                     
* - model    
  - str  
  - N/A  
  - Lidar model  
* - site    
  - str  
  - N/A
  - Site identifier  
* - z_id    
  - str  
  - N/A
  - Instrument identifier (e.g., 01, 02, etc.)
* - data_level_out
  - str  
  - N/A
  - Output data level (generally a0)    
* - save_file
  - boolean  
  - N/A
  - Whether to save file 
* - save_path
  - str  
  - N/A
  - Location where file is saved. If None, the current path is used
* - replace
  - boolean  
  - N/A
  - Whether to replace existing file. If False, existing files are skipped  

```

LiDAR models that currently are supported are:
* Streamline Halo XR and XR+ that produce .hpl files
* WindCube 200S that produce netCDF files

New models can be added by creating dedicated functions inside the LIDARGO_format class.