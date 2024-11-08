'''
Test LIDARGO
'''

import os
cd=os.path.dirname(__file__)
import glob
import LIDARGO_standardize as lb0
import LIDARGO_statistics as lc0

files=glob.glob('data/propietary/awaken/volumetric-wake-csm/*a0*nc')#add wildcard name of formatted files to process
for f in files:
    lproc = lb0.LIDARGO(f,'config/config_awaken_b0.xlsx', verbose=True)
    lproc.process_scan(f,replace=False,save_file=True)

files=glob.glob('data/propietary/awaken/volumetric-wake-csm/*b0*nc')#add wildcard name of standardized files to process
for f in files:
    lproc = lc0.LIDARGO(f,'config/config_awaken_c0.xlsx', verbose=True)
    lproc.process_scan(f,replace=True,save_file=True)