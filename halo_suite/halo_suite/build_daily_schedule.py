'''
Build daily scan schedule
'''

import os
cd=os.path.dirname(__file__)
import datetime as dt
import numpy as np
from tkinter import Tk
from halo_suite.utilities import scan_file_compiler
from tkinter.filedialog import askopenfilename
import yaml
import pandas as pd

root = Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

#%% Inputs
source=filename = askopenfilename(
    title="Scan schedule file",
    filetypes=[("All files", "*.xlsx")],
    initialdir=cd,
)
assert os.path.isfile(source), f'Invalid file "{source}"' 

path_config_lidar=filename = askopenfilename(
    title="Lidar configuration file",
    filetypes=[("All files", "*.yaml")],
    initialdir=cd,
)
assert os.path.isfile(path_config_lidar),  f'Invalid file "{path_config_lidar}"' 

time_corr=np.float64(input('Buffer time [%]: '))

#%% Initialization

#read schedule specs
scan_info=pd.read_excel(source) 
filenames=scan_info['Scan file'].values
T=scan_info['Total time [s]'].values
tau_s=scan_info['Sampling time [s]'].values
avg_loops=scan_info['Average loops (SSM only)'].values

#read lidar config
with open(path_config_lidar, 'r') as fid:
    config_lidar = yaml.safe_load(fid)  

#calculat reps
N=np.floor(T/(tau_s*(1+time_corr/100))).astype(int)

#add reps and total time
scan_info['Reps']=N
scan_info['Total time (real)']=tau_s*N

#file system
save_path=os.path.join(os.path.dirname(source),os.path.basename(source)[:-5]+'.dss')
os.makedirs(os.path.dirname(save_path),exist_ok=True)

#zeroing
scan_files=[]

#%% Main
for f,n in zip(filenames,N):
    a1=np.float64(f.split('_')[0])
    ra=f.split('_')[1]
    a2=np.float64(f.split('_')[2])
    e1=np.float64(f.split('_')[3])
    re=f.split('_')[4]
    e2=np.float64(f.split('_')[5])
    mode_ppr=f.split('_')[6]
    volumetric='vol' in f.split('_')[7]
    reps=int(f.split('x')[1].split('.')[0])
    
    if mode_ppr=='ssm':
        mode='ssm'
    elif mode_ppr[:3] =='csm':
        mode='csm'
        try:
            ppr=int(mode_ppr[3:])
        except:
            raise ValueError(f'Could not parse PPR from filename "{f}"')
            
        avg_loops=np.zeros(len(avg_loops))+ppr/1000
        
    if '.' in ra and '.' in re:
        ra=np.float64(ra)
        re=np.float64(re)
        res_mode='degrees'
    elif '.' not in ra and '.' not in re:
        ra=int(ra)
        re=int(re)
        res_mode='count'
    else:
        raise ValueError(f'Could not parse angular resolution from filename "{f}"')
        
    if res_mode=='degrees':
        ra+=10**-10
        re+=10**-10
        azi=np.arange(a1,a2+ra/2,ra)
        ele=np.arange(e1,e2+re/2,re)
        identifier=f'{a1:.2f}_{ra:.2f}_{a2:.2f}_{e1:.2f}_{re:.2f}_{e2:.2f}'
    elif res_mode=='count':
        azi=np.linspace(a1,a2,ra)
        ele=np.linspace(e1,e2,re)
        identifier=f'{a1:.2f}_{ra}_{a2:.2f}_{e1:.2f}_{re}_{e2:.2f}'
    
    if mode=='ssm':
        scan_file=scan_file_compiler(mode=mode.upper(),azi=azi,ele=ele,repeats=n*reps,
                                     identifier=identifier,save_path=os.path.dirname(save_path),
                                     volumetric=volumetric,reset=False)
    elif mode=='csm':
        scan_file=scan_file_compiler(mode=mode.upper(),azi=azi,ele=ele,repeats=n*reps,ppr=ppr,
                            identifier=identifier,save_path=os.path.dirname(save_path),
                            config=config_lidar,
                            optimize=True,volumetric=volumetric,reset=False)
        scan_files.append(os.path.basename(scan_file))
    
N_day=np.floor(24*3600/np.sum(T)).astype(int)
s=''
for i_seq in range(N_day):
    for i_scan in range(len(T)):
        t=np.float64(i_seq*np.sum(T)+np.sum(T[:i_scan]))
        
        t_str=dt.datetime.strftime(dt.datetime(2000, 1, 1)+dt.timedelta(seconds=t),'%H%M%S')
        f=os.path.basename(filenames[i_scan][:-4])+'x'+str(N[i_scan])
        s+=t_str+'\t'+scan_files[i_scan]+'\t'+str(int(avg_loops[i_scan]))+'\t'+mode[0]+'\t7\n'
        
with open(save_path,'w') as fid:
    fid.write(s[:-1])
    
scan_info.set_index('Scan name').to_excel(source)