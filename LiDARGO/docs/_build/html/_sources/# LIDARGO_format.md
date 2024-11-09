# LIDARGO_format overview

LIDARGO_format.py is a class that converts raw lidar data files into netCDF datasets compatible with LIDARGO_standardize.py and compliant with the [Wind Data Hub](https://a2e.energy.gov/login) standards.

The class initialization requires the folling inputs:
```{list-table} Inputs to LIDARGO_format.__init__.
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

The main module is the process_scan which accepts the following inputs:

```{list-table} Inputs to LIDARGO_format.process_scan.
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