import pandas as pd
import lidargo as lg

# make dictionary out of the configuration data
source_config_stand = "../config/config_raaw_b1.xlsx"
config = pd.read_excel(source_config_stand)
config = config.set_index("PARAMETER").T
config.index = config.name
config = config.T["wake.turb"].to_dict()

### Instantiate a configuration object
config = lg.LidarConfig(**config)

exampleFilePath = "nrel.lidar.z02.b0.20230407.073004.user5.nc"

### This is the whole show:
# instatiate the object with a file path, a config object, and the verbose flag
b0 = lg.Standardize(exampleFilePath, config=config, verbose=True)
# process the scan data (standardize coordinates and conduct QC)
b0.process_scan(replace=True, save_file=True, make_figures=True, save_path=".")
# either use the make_figures flag or call the qc_report method separately
# b0.qc_report(saveFigs=True)
