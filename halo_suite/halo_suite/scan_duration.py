# -*- coding: utf-8 -*-
'''
Calculate scan duration from raw data files
'''
import os
cd=os.path.dirname(__file__)

import numpy as np
import linecache
import glob 
import yaml
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

plt.close('all')

#%% Inputs
path_config=os.path.abspath('configs/config.yaml')
date=input('Date (YYYYmmdd): ')
time_res=np.float64(input('Time resolution [min]: '))

#%% Initialization
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)  

if "Proc" in config['path_data']:
    source=os.path.join(date[:4],date[:6],date)
else:
    source=''

files=sorted(glob.glob(os.path.join(config['path_data'],source,'*.hpl')))
assert len(files)>0, f"No files found on {date}"

#get lidar ID
lidar_id=int(os.path.basename(files[0]).split('_')[1])

#filesystem
os.makedirs(os.path.abspath('schedules'),exist_ok=True)

#zeroing
t1=[]
t2=[]
duration=[]
filename=[]
scan_name=[]
date_dt=datetime.datetime.strptime(date, '%Y%m%d')

#%% Main
for f in files:
    if date in f:
        try:
            N_tot = sum(1 for line in open(f))
            Nr=int(linecache.getline(f, 3).split(':')[1])
            h1=np.float64(linecache.getline(f, 18).split(' ')[0])
            h2=np.float64(linecache.getline(f, N_tot-Nr).split(' ')[0])
            t1=np.append(t1,date_dt+datetime.timedelta(hours=h1))
            t2=np.append(t2,date_dt+datetime.timedelta(hours=h2)+datetime.timedelta(seconds=1))
            duration=np.append(duration,(h2-h1)*3600)
            scan_name=np.append(scan_name,linecache.getline(f, 8).split(':')[1].split(' - ')[0].strip())
            filename=np.append(filename,f)
        
            linecache.clearcache()
            print(os.path.basename(f)+' done')
        except Exception as e:
            print(e)

#calculate min/max duration
duration_min={}
duration_max={}
for s in np.unique(scan_name):
    duration_min[str(s)]=np.round(duration[scan_name==s].min(),1)
    duration_max[str(s)]=np.round(duration[scan_name==s].max(),1)
    
#%% Ouput
Output=pd.DataFrame()
Output['Filename']=filename
Output['Scan name']=scan_name
Output['Start time']=t1
Output['End time']=t2
Output['Duration [s]']=duration

Output.to_excel(os.path.abspath('schedules/'+date+'_'+str(lidar_id)+'_schedule.xlsx'))

#%% Plots
#scan sequence
sel=scan_name!='Stare'
predefined_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
N=len(np.unique(scan_name[sel]))
cols = predefined_colors[:N]
                                    
plt.figure(figsize=(18,6))

plt.subplot(2,1,1)
t0=np.arange(date_dt,t2[-1],datetime.timedelta(minutes=time_res))
time1=mdates.date2num(t1[sel])
time2=mdates.date2num(t2[sel])
time0=mdates.date2num(t0)

for tt1,tt2,s in zip(time1,time2,scan_name[sel]):
    i_scan=np.where(s==np.unique(scan_name[sel]))[0][0]
    plt.barh(0,tt2-tt1,left=tt1,label=f'{s}: {duration_min[s]} -  {duration_max[s]} s',color=cols[i_scan])
    
for tt0 in time0:
    plt.plot([tt0,tt0],[-0.5,0.5],'--k')

plt.gca().xaxis_date()
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = list(set(labels))
unique_handles = [handles[labels.index(label)] for label in unique_labels]
plt.legend(unique_handles, unique_labels)
plt.xlabel('UTC time')
plt.grid()
plt.yticks([0,1],labels=[])
plt.ylim([-0.5,0.5])
plt.xlim(mdates.date2num(t1.min())-5/24/60,mdates.date2num(t2.max())+5/24/60)

plt.subplot(2,1,2)
sel=scan_name=='Stare'
t0=np.arange(date_dt,t2[-1],datetime.timedelta(minutes=time_res))
time1=mdates.date2num(t1[sel])
time2=mdates.date2num(t2[sel])
time0=mdates.date2num(t0)
ctr=0
for tt1,tt2,s in zip(time1,time2,scan_name[sel]):
    plt.barh(0,tt2-tt1,left=tt1,color='y')
    ctr+=1

for tt0 in time0:
    plt.plot([tt0,tt0],[-0.5,0.5],'--k')

plt.gca().xaxis_date()
plt.xlabel('UTC time')
plt.grid()
plt.yticks([0,1],labels=[])
plt.ylim([-0.5,0.5])
plt.xlim(mdates.date2num(t1.min())-5/24/60,mdates.date2num(t2.max())+5/24/60)
plt.title('Stare files')
plt.tight_layout()

plt.savefig(os.path.abspath('schedules/'+date+'_'+str(lidar_id)+'_schedule.png'))


    