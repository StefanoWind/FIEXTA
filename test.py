'''
Test LIDARGO
'''

import os
cd=os.path.dirname(__file__)
import glob
import LIDARGO_standardize as lb0
import LIDARGO_statistics as lc0

files=glob.glob('data/**/*a0*nc')
for f in files:
    lproc = lb0.LIDARGO(f,'data/configs_standardize.xlsx', verbose=True)
    lproc.process_scan(f,replace=True,save_file=True)

files=glob.glob('data/**/*b0*nc')
for f in files:
    lproc = lc0.LIDARGO(f,'data/configs_statistics.xlsx', verbose=True)
    lproc.process_scan(f,replace=True,save_file=True)