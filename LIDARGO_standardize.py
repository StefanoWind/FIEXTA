import os
import xarray as xr
import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
import re
import matplotlib
import socket
import getpass
from datetime import datetime
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import sys

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
warnings.filterwarnings("ignore")

class LIDARGO:
    def __init__(self, source, config_file, verbose=True,logger=None):
        '''
        Initialize the LIDAR data processor with configuration parameters.

        Inputs:
        ------
        source: str
            The input-level file name 
        config_file: str
            The configuration file
        verbose: bool
            Whether or not print QC-related information
        '''
        plt.close('all')
        
        self.source=source
        self.verbose=verbose
        self.logger=logger
        
        self.print_and_log(f'Initializing standardization of {os.path.basename(self.source)}')
        
        #load configuration
        configs=pd.read_excel(config_file).set_index('PARAMETER')
        date_source=np.int64(re.search(r'\d{8}.\d{6}',source).group(0)[:8])
        matches=[]
        for regex in configs.columns:
            match = re.findall(regex, source)
            sdate=configs[regex]['start_date']
            edate=configs[regex]['end_date']
            if len(match)>0 and date_source>=sdate and date_source<=edate:
                matches.append(regex)
        
        if len(matches)==0:
            self.print_and_log('No regular expression matching the file name')
            return
        elif len(matches)>1:
            self.print_and_log('Mulitple regular expressions matching the file name')
            return
                
        config=configs[matches[0]].to_dict()
        self.config = config
        
        #Dynamically assign attributes from the dictionary
        for key, value in config.items():
            setattr(self, key, value)
        
        #Load data
        self.inputData = xr.open_dataset(self.source)
       
        
    def print_and_log(self,message):
        if self.verbose:
            print(message)
        if self.logger is not None:
            self.logger.info(message)
        
    def check_data(self):
        '''
        Check input data for consistency
        '''

        # Check distance (range) array. 
        if 'range_gate' in self.inputData.coords:
            if 'overlapping' in self.inputData.attrs['Scan type']:
                distance = np.unique(self.inputData['distance_overlapped'])
            else:
                distance = np.unique(self.inputData['distance'])
            distance = distance[~np.isnan(distance)]
            if len(distance) == 0:
                self.print_and_log(f'WARNING: All distance values are invalid on {os.path.basename(self.source)}, skipping it')
                return False

        # Check for valid radial wind speed values
        if self.inputData['wind_speed'].isnull().sum() == self.inputData['wind_speed'].size:
            # All wind speeds are NaNs. Skipping
            self.print_and_log(f'WARNING: All wind speeds are invalid on {os.path.basename(self.source)}, skipping it')
            return False

        # Check for valid azimuth angles
        if self.inputData['azimuth'].isnull().sum() == self.inputData['azimuth'].size:
            # All azimuths are NaNs. Skipping
            self.print_and_log(f'WARNING: All azimuths are invalid on {os.path.basename(self.source)}, skipping it')
            return False
        
        # Check for valid elevation angles
        if self.inputData['elevation'].isnull().sum() == self.inputData['elevation'].size:
            # All azimuths are NaNs. Skipping
            self.print_and_log(f'WARNING: All elevations are invalid on {os.path.basename(self.source)}, skipping it')
            return False

        return True

    def process_scan(self, make_figures=True, save_file=True, save_path=None, replace=True):
        '''
        Run all the processing.
    
        Inputs:
        -------
        make_figures: bool
            Whether or not generate QC figures
        save_file: bool
            Whether or not save the final processed file
        save_path: str
            String with the directory the destination of the processed files.
            Optional, defaults to analogous input, replacing input-level with output-level
            data. Creates the necessary intermediate directories
        replace: bool
            Whether or not to replace processed scan if one already exists
        '''
        
        # Check if file has been processed yet and whether is to be replaced
        if 'config' not in dir(self):
            self.print_and_log(f'No configuration available. Skipping file {os.path.basename(self.source)}')
            return
        
        #Compose filename
        save_filename = self.data_level_out.join(self.source.split(self.data_level_in)).replace('.nc','.'+self.project+'.'+self.name+'.nc')
        if save_path is not None:
            save_filename=os.path.join(save_path,os.path.basename(save_filename))
        
        self.save_filename=save_filename
        
        if os.path.exists(os.path.dirname(save_filename))==False:
            os.makedirs(os.path.dirname(save_filename))
            
        if save_file and not replace and os.path.isfile(save_filename):
            self.print_and_log(f'Processed file {save_filename} already exists, skipping it')
            return
        else:
            self.print_and_log(f'Generating standardized file {os.path.basename(save_filename)}')
    
        # Check data
        if not self.check_data():
            return
       
        # Filter pre-processing
        self.remove_back_swipe()
        self.bin_and_count_angles()
        self.update_angles_to_nominal()
        
        # Apply qc filter
        self.filter_scan_data()
        
        # Re-index scan
        self.calculate_repetition_number()
        self.calculate_beam_number()
        self.identify_scan_mode()
        self.reindex_scan()
        self.add_attributes()
    
        if make_figures:
            self.qc_report()
    
        if save_file:
            self.outputData.to_netcdf(save_filename)
            self.print_and_log(f'Standardized file saved as {save_filename}')

    def remove_back_swipe(self):
        
        '''
        Reject fast scanning head repositioning, identified based on the azimuth and elevation step thresholds.

        '''
        
        #Angular difference (forward difference)
        diff_azi_fw=self.inputData['azimuth'].diff(dim='time',label='lower')
        diff_azi_fw[diff_azi_fw>180]= diff_azi_fw[diff_azi_fw>180]-360
        diff_azi_fw[diff_azi_fw<-180]=diff_azi_fw[diff_azi_fw<-180]+360
        
        diff_ele_fw=self.inputData['elevation'].diff(dim='time',label='lower')
        diff_ele_fw[diff_ele_fw>180]= diff_ele_fw[diff_ele_fw>180]-360
        diff_ele_fw[diff_ele_fw<-180]=diff_ele_fw[diff_ele_fw<-180]+360
        
        forward_swipe_condition_fw=(diff_azi_fw>=self.min_azi_step)*(diff_azi_fw<=self.max_azi_step)*\
                                   (diff_ele_fw>=self.min_ele_step)*(diff_ele_fw<=self.max_ele_step)
        
        #Angular difference (backward difference)
        diff_azi_bw=self.inputData['azimuth'].diff(dim='time',label='upper')
        diff_azi_bw[diff_azi_bw>180]= diff_azi_bw[diff_azi_bw>180]-360
        diff_azi_bw[diff_azi_bw<-180]=diff_azi_bw[diff_azi_bw<-180]+360
        
        diff_ele_bw=self.inputData['elevation'].diff(dim='time',label='upper')
        diff_ele_bw[diff_ele_bw>180]= diff_ele_bw[diff_ele_bw>180]-360
        diff_ele_bw[diff_ele_bw<-180]=diff_ele_bw[diff_ele_bw<-180]+360
                     
        forward_swipe_condition_bw=(diff_azi_bw>=self.min_azi_step)*(diff_azi_bw<=self.max_azi_step)*\
                                   (diff_ele_bw>=self.min_ele_step)*(diff_ele_bw<=self.max_ele_step)
                                   
        #Contatenate swipe conditions
        forward_swipe_condition=xr.concat([forward_swipe_condition_fw[0],
                                           forward_swipe_condition_fw|forward_swipe_condition_bw,
                                           forward_swipe_condition_bw[-1]],dim='time')
        
        #Remove beams in the deceleration phase of the scanning head
        azi_sel=self.inputData['azimuth'].where(forward_swipe_condition)
        ele_sel=self.inputData['elevation'].where(forward_swipe_condition)
        first=(azi_sel-azi_sel[0])**2+(ele_sel-ele_sel[0])**2<self.ang_tol**2
        first_forward=xr.concat([first[0],(forward_swipe_condition+0).diff(dim='time')==1],dim='time')
        forward_swipe_condition[(first_forward==True)*(first==False)]=False
        
        self.outputData = self.inputData.where(forward_swipe_condition)
        self.azimuth_selected=self.outputData['azimuth'].copy()
        self.elevation_selected=self.outputData['elevation'].copy()
        
        self.print_and_log(f'Back-swipe removal: {np.round(np.sum(forward_swipe_condition).values/len(self.inputData.azimuth)*100,2)}% retained')

    def bin_and_count_angles(self):
        '''
        Perform binning and counting of scan data to identify most probable angles.
        '''
        from scipy import stats
        
        llimit_azi = floor(self.outputData.azimuth.min(), self.ang_tol) - self.ang_tol / 2
        ulimit_azi = ceil(self.outputData.azimuth.max(), self.ang_tol) + self.ang_tol 
        azi_bins = np.arange(llimit_azi, ulimit_azi, self.ang_tol)
        
        llimit_ele = floor(self.outputData.elevation.min(), self.ang_tol) - self.ang_tol / 2
        ulimit_ele = ceil(self.outputData.elevation.max(), self.ang_tol) + self.ang_tol 
        ele_bins = np.arange(llimit_ele, ulimit_ele, self.ang_tol)
        
        E,A=np.meshgrid(mid(ele_bins),mid(azi_bins))
        counts=stats.binned_statistic_2d(self.outputData.azimuth, self.outputData.elevation,None, 'count', bins=[azi_bins, ele_bins])[0]       
        counts_condition=counts/counts.max()> self.count_threshold
        
        self.counts = counts[counts_condition]
        self.azimuth_detected=A[counts_condition]
        self.elevation_detected=E[counts_condition]
          
    def update_angles_to_nominal(self):
        '''
        Update azimuth values to the nearest nominal bin centers within the specified tolerance.

        '''

        if self.outputData is not None and 'azimuth' in self.outputData and 'elevation' in self.outputData:
            azi = self.outputData['azimuth'].values
            azimuth_bin_centers = self.azimuth_detected
            ele = self.outputData['elevation'].values
            elevation_bin_centers = self.elevation_detected

            # Calculate the absolute difference between azimuth values and bin centers
            diff_ang = (np.abs(azi[:, None] - azimuth_bin_centers[None, :])**2+\
                        np.abs(ele[:, None] - elevation_bin_centers[None, :])**2)**0.5
            mindiff = xr.DataArray(data=diff_ang.min(axis=1), coords={'time': self.outputData.time})
            mindiff[mindiff>self.ang_tol]=np.nan
            minind = np.argmin(diff_ang, axis=1)
            
            self.outputData['azimuth'].values = azimuth_bin_centers[minind]
            self.outputData['elevation'].values = elevation_bin_centers[minind]

            self.outputData = self.outputData.where(~np.isnan(mindiff))
            self.azimuth_regularized=self.outputData['azimuth'].copy()
            self.elevation_regularized=self.outputData['elevation'].copy()
            
            self.print_and_log(f'Relevant angles detection: {np.round(np.sum(~np.isnan(self.azimuth_regularized.values+self.elevation_regularized.values))/len(self.inputData.azimuth)*100,2)}% retained')
          
        else:
            raise ValueError('Dataset is not initialized or does not contain angle values')

    
    def filter_scan_data(self):
        '''
        QC lidar data

        '''
        
        df = self.scan_to_dataframe()
        
        #Apply prefiltering
        filterdf1=self.pre_filter(df)
        df_temp=df.where(filterdf1.sum(axis=1)==len(filterdf1.columns))
        
        #Apply dynamic filter
        filterdf2,rws_norm,snr_norm,probability = self.dynamic_filter(df_temp)
    
        #Save qc flags
        filterdf=pd.concat([filterdf1,filterdf2],axis=1)
        df['qc_wind_speed']=0
        self.qc_flag={}
        ctr=1
        for c in filterdf.columns:
            self.qc_flag[c]=ctr
            if np.sum(filterdf[c]==False)>0:
                failed=np.where(filterdf[c].values==False)[0]
                record=(df['qc_wind_speed'].iloc[failed]==0).values
                df['qc_wind_speed'].iloc[failed[record]]=ctr
            ctr+=1

        #Reorganize output dataframe
        df['rws_norm']=rws_norm
        df['snr_norm']=snr_norm
        df['probability']=probability
        df = df.drop(['deltaTime'], axis=1)
        df = df[~df.index.duplicated(keep='first')]
        df = df.set_index(['time','range'])
        
        #Back to xarray
        ds = xr.Dataset.from_dataframe(df)
        
        #Drop range dimension from beam properties
        ds['azimuth']=  ds['azimuth'].sel(range=ds['range'][0]).drop('range')
        ds['elevation']=ds['elevation'].sel(range=ds['range'][0]).drop('range')
        ds['pitch']=  ds['pitch'].sel(range=ds['range'][0]).drop('range')
        ds['roll']=ds['roll'].sel(range=ds['range'][0]).drop('range')
        
        #Inherit attributes
        ds.attrs=self.outputData.attrs
        
        for v in self.outputData.var():
            if v in ds.var():
                ds[v].attrs=self.outputData[v].attrs
            
        #Save filtered data
        self.outputData = ds
        
    def scan_to_dataframe(self):
        
        '''
        Make a pandas dataframe to simplify filtering
        
        '''
        
        #Add SNR floor
        self.outputData['SNR']=self.outputData['SNR'].fillna(self.snr_min-1)
        
        #Add time in seconds from start of the scan
        tnum=(self.outputData['time'] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        self.outputData['deltaTime']=tnum-tnum.min()
            
        #Swap range index with physical range
        if 'overlapping' in self.inputData.attrs['Scan type']:
            distance = np.unique(self.outputData.distance_overlapped)
        else:
            distance = np.unique(self.outputData.distance)
        distance = distance[~np.isnan(distance)]
        self.outputData = self.outputData.rename({'range_gate': 'range'})
        self.outputData = self.outputData.assign_coords({'range': distance})

        #Add 3D coordinates
        self.outputData['x'],self.outputData['y'],self.outputData['z']=lidar_xyz(self.outputData['range'],self.outputData['elevation'],self.outputData['azimuth']+self.azimuth_offset)
        
        df = self.outputData[['x','y','z','wind_speed','SNR','deltaTime','azimuth','elevation','pitch','roll']].to_dataframe().dropna(how='any')
        df = df.reset_index()
        
        return df
    
    def pre_filter(self,df):
        '''
        Pre-filter of lidar data based on location and static rws and snr limits
        
        Inputs: 
        ------
        df: dataframe
            dataframe of lidar data
            
        Outputs:
        ------    
        filterdf: dataframe
            dataframe of QC flags
        df['rws_norm']: array of floats
            normalized radial wind speed
        df['snr_norm']: array of floats
            normalized SNR
        df['probability']: arrays of floats
            probability of rws-SNR histogram        
        '''
        filterdf=pd.DataFrame()
        
        #Range limits
        filt=(df['range']>=self.range_min)&(df['range']<=self.range_max)
        self.print_and_log(f'range_limit filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['range_limit'] = filt.values
        
        #Ground rejection
        filt=df['z']>self.ground_level
        self.print_and_log(f'ground_limit filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['ground_limit'] = filt.values
        
        #SNR limit
        filt=df['SNR']>=self.snr_min
        self.print_and_log(f'snr_limit filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['snr_limit'] = filt
        
        #Wind speed max limit
        filt=np.abs(df['wind_speed'])<=self.rws_max
        self.print_and_log(f'rws_max filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['rws_max'] = filt

        #Wind speed min limit
        df_temp=df.where(filterdf.sum(axis=1)<len(filterdf.columns))
        self.rws_min=self.detect_resonance(df_temp)
        
        filt=np.abs(df['wind_speed'])>=self.rws_min
        self.print_and_log(f'rws_min filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['rws_min'] = filt
     
        return filterdf
    
    def detect_resonance(self,df):
        '''
        Detect presence of resonance of bad data around 0 (some lidars have outliers clusterees aroun 0 m/s instead of uniformly spread acorss the bandwidth)
        
        Inputs:
        -----
        df: dataframe
            Dataframe of current data (bad data only)
            
        Outputs
        -----
        rws_min: float
            Lower threshold to be applied to absolute value of RWS. Detection is based on Gaussian fitting of histogram.

        '''
        
        #build histogram if RWS normalized by maximum value 
        df['rws_bins']=pd.cut(df['wind_speed']/df['wind_speed'].max(), bins=np.linspace(-1,1,self.N_resonance_bins))
        groups = df.groupby('rws_bins',group_keys=False)
        H=groups['wind_speed'].count()
        
        #Normalize histogram by subtracting min and dividing by value in 0 (makes it more Gaussian)
        H=(H-H.min())/H.iloc[int(self.N_resonance_bins/2-1)]
        
        #Single-parameter Gaussian fit
        H_x=np.array([x.mid for x in H.index])
        sigma=curve_fit(gaussian, H_x,H, p0=[0.1],bounds=[0,1])[0][0]
        
        #Check Gaussiainity and possibly calculate rws_min
        rmse=np.nanmean((gaussian(H_x,sigma)-H)**2)**0.5
        if rmse<=self.max_resonance_rmse:
            rws_min=2*sigma*df['wind_speed'].max()
            self.print_and_log('Detected resonance')
        else:
            rws_min=0
        
        self.config['rws_min']=rws_min
        return rws_min
    
    def dynamic_filter(self,df):
        '''
        Dynamic filter of lidar data based on normalized rws and snr based on Beck and Kuhn, Remote Sensing, 2017
        
        Inputs:
        -----
        df: dataframe
            dataframe of lidar data
        
        Outputs:
        -----
        filterdf: dataframe
            dataframe of QC flags
        
        '''
        filterdf=pd.DataFrame()
        
        #Group df by x, y, z, time bins
        df = defineLocalBins(df, self.config)
        groups = df.groupby(['xbins', 'ybins', 'zbins','timebins'],group_keys=False)

        #Normalized wind speed and SNR data channels
        df['rws_norm'] = groups['wind_speed'].apply(lambda x: x - x.median())
        df['snr_norm'] = groups['SNR'].apply(lambda x: x - x.median())
        
        #Normalized wind speed limit
        filt = np.abs(df['rws_norm']) <= self.rws_norm_limit
        self.print_and_log(f'rws_norm_limit filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['rws_norm_limit'] =  filt
            
        #Minimum population
        filt = groups['rws_norm'].count()>=self.local_population_min_limit
        temp = df.set_index(['xbins', 'ybins','zbins','timebins']).copy()
        temp['local_population_min_limit'] = filt
        filt = temp['local_population_min_limit'].reset_index(drop=True)
        self.print_and_log(f'local_population_min_limit filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['local_population_min_limit'] = filt

        #Standard error on the median filter
        filt = groups['rws_norm'].apply(
            lambda x: ((np.pi / 2) ** 0.5 * x.std()) / x.count() ** 0.5
            <= self.rws_standard_error_limit
        )
        temp = df.set_index(['xbins', 'ybins','zbins','timebins']).copy()
        temp['rws_standard_error_limit'] = filt
        filt = temp['rws_standard_error_limit'].reset_index(drop=True)
        self.print_and_log(f'rws_standard_error_limit filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['rws_standard_error_limit'] = filt

        filt = groups['snr_norm'].apply(
            lambda x: ((np.pi / 2) ** 0.5 * x.std()) / x.count() ** 0.5
            <= self.snr_standard_error_limit
        )
        temp = df.set_index(['xbins', 'ybins','zbins','timebins']).copy()
        temp['snr_standard_error_limit'] = filt
        filt = temp['snr_standard_error_limit'].reset_index(drop=True)
        self.print_and_log(f'snr_standard_error_limit filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['snr_standard_error_limit'] = filt

        #Probability conditions (applies actual dynamic filter)
        df['filtered_temp']=filterdf.sum(axis=1)==len(filterdf.columns)#points retained in previous steps
        filt,df,rws_range,probability_threshold = local_probability(df, self.config)
        self.print_and_log(f'probability_limit filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['probability_limit'] = filt
        self.qc_rws_range=rws_range
        self.qc_probability_threshold=probability_threshold

        #Local scattering filter (removes isolated points)   
        df['filtered_temp']=filterdf.sum(axis=1)==len(filterdf.columns)#points retained in previous steps
        filt = groups['filtered_temp'].mean() > self.local_scattering_min_limit
        temp = df.set_index(['xbins', 'ybins','zbins','timebins']).copy()
        temp['local_scattering_min_limit'] = filt
        filt = temp['local_scattering_min_limit'].reset_index(drop=True)
        filt = filt.reset_index(drop=True)
        self.print_and_log(f'local_scattering_min_limit filter: {np.round(filt.sum()/len(filt)*100,2)}% retained')
        filterdf['local_scattering_min_limit' ] = filt
        df=df.drop('filtered_temp', axis=1)
            
        filterdf = filterdf.replace(np.nan, False).astype(np.int8)
        self.print_and_log(f'Retained {np.round(100*filterdf.all(axis=1).sum()/len(filterdf),2)}% of data after QC')

        return filterdf,df['rws_norm'],df['snr_norm'],df['probability']
  

    def calculate_repetition_number(self):
        '''
        Calculate the repetition number of the scan
        '''
        azi = self.outputData.azimuth.values
        ele = self.outputData.elevation.values
        time= self.outputData.time.values

        #Find scans' start
        scan_start = time[(azi == azi[0])*(ele==ele[0])]
        
        #remove short scans (idling beam)
        scan_duration=np.diff(np.append(scan_start,time[-1]))
        scanID = pd.Series(data=np.zeros(len(scan_start))-9999, index=scan_start).rename_axis('time')
        duration_condition=pd.Series(data=scan_duration>np.timedelta64(int(self.min_scan_duration*10**9),'ns'), index=scanID.index)
        scanID[duration_condition]=np.arange(np.sum(duration_condition))
        scan_start_time=pd.Series(data=scan_start, index=scanID.index).rename_axis('time')
        
        #add scanID to dataset 
        self.outputData['scanID'] = scanID
        self.outputData['scanID'].values = self.outputData['scanID'].ffill(dim='time')
        self.outputData['scan_start_time'] = scan_start_time
        self.outputData['scan_start_time'].values = self.outputData['scan_start_time'].ffill(dim='time')
        self.outputData=self.outputData.where(self.outputData['scanID']>=0)
        
    def calculate_beam_number(self):
        '''
        Calculate the beam number based on how long after each scan start an angular position occurs on average
        '''
        self.outputData['deltaTime']=(self.outputData.time-self.outputData.scan_start_time)/np.timedelta64(1, 's')
        deltaTime_regularized=np.zeros(len(self.outputData.time))+np.nan
        deltaTime_median=np.zeros(len(self.azimuth_detected))
        ctr=0
        for a,e in zip(self.azimuth_detected,self.elevation_detected):
            sel=(self.outputData.azimuth.values==a)*(self.outputData.elevation.values==e)
            deltaTime_median[ctr]=np.nanmedian(self.outputData.deltaTime[sel])
            deltaTime_regularized[sel]=np.nanmedian(self.outputData.deltaTime[sel])
            ctr+=1
            
        sort_angles=np.argsort(deltaTime_median[~np.isnan(deltaTime_median)])
        beamID=np.zeros(len(deltaTime_regularized))+np.nan
        beamID[~np.isnan(deltaTime_regularized)]=np.where((deltaTime_regularized[:,None]-np.unique(deltaTime_regularized)[None,:])==0)[1]
        
        self.azimuth_detected=self.azimuth_detected[~np.isnan(deltaTime_median)][sort_angles]
        self.elevation_detected=self.elevation_detected[~np.isnan(deltaTime_median)][sort_angles]
        self.counts=self.counts[~np.isnan(deltaTime_median)][sort_angles]
        self.outputData['beamID']=xr.DataArray(data=beamID,coords={'time':self.outputData.time.values})
        
    def reindex_scan(self):
        '''
        Reindex the dataset from [time, range] to [range, beamID, scanID].

        Keeps all the time information in the dataset.
        '''
        if self.outputData is not None:
            
            #Copy time information
            self.outputData['time_temp']=xr.DataArray(self.outputData.time.values,coords={'time':self.outputData.time.values})
            
            #Reindex
            self.outputData=self.outputData.where(~np.isnan(self.outputData.beamID+self.outputData.scanID),drop=True)
            self.outputData = self.outputData.set_index(time=['beamID', 'scanID'])
            self.outputData = self.outputData.drop_duplicates('time').unstack()
          
            #Finalize dataset
            self.outputData= self.outputData.rename_vars({'time_temp':'time'})
            self.outputData=self.outputData.drop_vars(['scan_start_time','deltaTime'])
            
        else:
            raise ValueError('Dataset is not initialized')
    
    def identify_scan_mode(self):
        '''
        Identify the type of scan, which is useful for plotting
        '''
        azi = self.outputData['azimuth'].values
        ele = self.outputData['elevation'].values
        
        azimuth_variation= np.abs(np.nanmax(np.tan(azi))-np.nanmin(np.tan(azi)))
        elevation_variation= np.abs(np.nanmax(np.cos(ele))-np.nanmin(np.cos(ele)))
        
        if elevation_variation<10**-10 and azimuth_variation<10**-10:
            self.outputData.attrs['scan_mode']='Stare'
        elif elevation_variation<10**-10 and azimuth_variation>10**-10:
            self.outputData.attrs['scan_mode']='PPI'
        elif elevation_variation>10**-10 and azimuth_variation<10**-10:
            self.outputData.attrs['scan_mode']='RHI'
        else:
            self.outputData.attrs['scan_mode']='3D'
            
    def add_attributes(self):
         
        for v in self.outputData.var():
            if 'ancillary_variables' in self.outputData[v].attrs:
                del self.outputData[v].attrs['ancillary_variables']
                
        for v in self.outputData.var():
            self.outputData[v].attrs['standard_name']=v
                
        self.outputData['range'].attrs['standard_name']='range'
        self.outputData['range'].attrs['long_name']='Range'
        self.outputData['range'].attrs['description']='Distance from the lidar source.'
        self.outputData['range'].attrs['units']='m'    
        
        self.outputData['beamID'].attrs['standard_name']='beamID'
        self.outputData['beamID'].attrs['long_name']='Beam ID'
        self.outputData['beamID'].attrs['description']='Index of the beam within a scan.'
        self.outputData['beamID'].attrs['units']='int'   
        
        self.outputData['scanID'].attrs['standard_name']='scanID'
        self.outputData['scanID'].attrs['long_name']='Scan ID'
        self.outputData['scanID'].attrs['description']='Repetition index of the scan.'
        self.outputData['scanID'].attrs['units']='int'    
        
        self.outputData['azimuth'].attrs['description']='Beam azimuth angle'
        self.outputData['elevation'].attrs['description']='Beam elevation angle'
        self.outputData['pitch'].attrs['description']='Lidar pitch angle'
        self.outputData['roll'].attrs['description']='Lidar roll angle'
                
        self.outputData['wind_speed'].attrs['long_name']='Line-of-sight velocity'
        self.outputData['wind_speed'].attrs['description']='Line-of-sight velocity.'
        self.outputData['wind_speed'].attrs['ancilary_variables']='qc_wind_speed'
        
        self.outputData['x'].attrs['long_name']='x-direction'
        self.outputData['x'].attrs['description']='x-direction.'
        self.outputData['x'].attrs['units']='m'
        
        self.outputData['y'].attrs['long_name']='y-direction'
        self.outputData['y'].attrs['description']='y-direction.'
        self.outputData['y'].attrs['units']='m'
        
        self.outputData['z'].attrs['long_name']='z-direction'
        self.outputData['z'].attrs['description']='z-direction.'
        self.outputData['z'].attrs['units']='m'
        
        self.outputData['rws_norm'].attrs['long_name']='Normalized radial wind speed'
        self.outputData['rws_norm'].attrs['description']='Fluctuation of radial wind speed on top of the binned spatio-temporal median. It is used in the dynamic filter.'
        self.outputData['rws_norm'].attrs['units']='m/s'
        
        self.outputData['snr_norm'].attrs['long_name']='Normalized SNR'
        self.outputData['snr_norm'].attrs['description']='Fluctuation of signal-to-noise ratio on top of the binned spatio-temporal median. It is used in the dynamic filter.'
        self.outputData['snr_norm'].attrs['units']='dB'
        
        self.outputData['probability'].attrs['long_name']='Probability'
        self.outputData['probability'].attrs['description']='Value of 2-D p.d.f. in the rws_norm vs snr_norm plane. It is used in the dynamic filter.'
        self.outputData['probability'].attrs['units']='no units' 
        
        self.outputData['time'].attrs['standard_name']='time'
        self.outputData['time'].attrs['long_name']='Time UTC'
        self.outputData['time'].attrs['description']='Timestamp in UTC format for the specific beam.'
        
        self.outputData['qc_wind_speed'].attrs=''
        self.outputData['qc_wind_speed'].attrs['units']='int' 
        self.outputData['qc_wind_speed'].attrs['long_name']='Wind speed QC flag'
        self.outputData['qc_wind_speed'].attrs['description']='This variable contains bit-packed integer values, where each bit represents a QC test on the data. Non-zero bits indicate the QC condition given in the description for those bits.'
        self.outputData['qc_wind_speed'].attrs['bit_0_description']='Value retained.'
        self.outputData['qc_wind_speed'].attrs['bit_0_assessment']='Good'
        for k in self.qc_flag.keys():
            self.outputData['qc_wind_speed'].attrs[f'bit_{self.qc_flag[k]}_description']='Value rejected due to '+k+' criterion.'
            self.outputData['qc_wind_speed'].attrs[f'bit_{self.qc_flag[k]}_assessment']='Bad'
            
        for k in self.config:
            self.outputData['qc_wind_speed'].attrs[k]=self.config[k]
            
        #Global attributes
        self.outputData.attrs['data_level']=self.data_level_out
        self.outputData.attrs['datastream']=os.path.basename(self.save_filename)
        self.outputData.attrs['contact']='stefano.letizia@nrel.gov'
        self.outputData.attrs['institution']='NREL'
        self.outputData.attrs['description']='AWAKEN Halo XR/XR+ Lidar standardized and quality-controlled data'
        self.outputData.attrs['history']='Generated by '+getpass.getuser()+' on '+socket.gethostname()+\
                ' on '+datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S')+' using '+os.path.basename(sys.argv[0])
        del self.outputData.attrs['code_version']
        
    def qc_report(self):
        '''
        Make figures.
        '''
        
        rws=self.outputData['wind_speed'].copy()
        rws_qc=self.outputData['wind_speed'].where(self.outputData.qc_wind_speed==0)
        X=self.outputData['x'].mean(dim='scanID')
        Y=self.outputData['y'].mean(dim='scanID')
        Z=self.outputData['z'].mean(dim='scanID')
        r=self.outputData['range']
        
        fig_probability=plt.figure(figsize=(18,8))
        prob_arange=np.arange(np.log10(self.min_probability_range)-1,np.log10(self.max_probability_range)+1)
        ax1=plt.subplot(1,2,1)
        sp=plt.scatter(self.outputData.rws_norm,self.outputData.snr_norm,s=3,c=np.log10(self.outputData.probability),cmap='hot',vmin=prob_arange[0],vmax=prob_arange[-1])
        plt.xlabel('Normalized radial wind speed [m s$^{-1}$]')
        plt.ylabel('Normalized SNR [dB]')
        plt.grid()
        plt.title('QC of data at '+self.inputData.attrs['location_id']+' on '+datestr(np.nanmean(dt64_to_num(self.inputData['time'])),'%Y-%m-%d')+\
                    '\n File: '+os.path.basename(self.source))
        
        ax2=plt.subplot(1,2,2)
        plt.semilogx(self.outputData.probability.values.ravel(),self.outputData.rws_norm.values.ravel(),'.k',alpha=0.01,label='All points')
        plt.semilogx([10**b.mid for b in self.qc_rws_range.index],self.qc_rws_range,'.b',markersize=10,label='Range')
        plt.semilogx([self.qc_probability_threshold,self.qc_probability_threshold],[-self.rws_norm_limit,self.rws_norm_limit],'--r',label='Threshold')
        plt.xlabel('Probability')
        plt.ylabel('Normalized radial wind speed [m s$^{-1}$]')
        ax2.yaxis.set_label_position('right')
        plt.legend()
        plt.grid()
        
        fig_probability.subplots_adjust(left=0.1, right=0.9, wspace=0.3)
        cax=fig_probability.add_axes([ax1.get_position().x0+ax1.get_position().width+0.01,ax1.get_position().y0,0.015,ax1.get_position().height])
        cbar = plt.colorbar(sp, cax=cax,label='Probability')
        cbar.set_ticks(prob_arange)
        cbar.set_ticklabels([r'$10^{'+str(int(p))+'}$' for p in prob_arange])
        
        N_plot=np.min(np.array([5,len(self.outputData.scanID)]))
        i_plot=[int(i) for i in np.linspace(0,len(self.outputData.scanID)-1,N_plot)]
        
        if self.outputData.attrs['scan_mode']=='Stare':
            time=(self.outputData.time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
            fig_rws=plt.figure(figsize=(18,9))
            ax=plt.subplot(2,1,1)
            pc=plt.pcolor(self.outputData['time'],r,rws[:,0,:],cmap='coolwarm',vmin=np.nanpercentile(rws_qc,5)-1,vmax=np.nanpercentile(rws_qc,95)+1)
            plt.xlabel('Time (UTC)')
            plt.ylabel(r'Range [m]')
            date_fmt = mdates.DateFormatter('%H:%M:%S')
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.grid()
            plt.title('Radial wind speed at '+self.inputData.attrs['location_id']+' on '+datestr(np.nanmean(time),'%Y-%m-%d')+\
                '\n File: '+os.path.basename(self.source)\
                      +'\n'+ datestr(np.nanmin(time),'%H:%M:%S')+' - '+datestr(np.nanmax(time),'%H:%M:%S'))
                
            cax=fig_rws.add_axes([ax.get_position().x0+ax.get_position().width+0.01,ax.get_position().y0,0.015,ax.get_position().height])
            cbar = plt.colorbar(pc, cax=cax,label='Raw radial \n'+r' wind speed [m s$^{-1}$]')
    
            ax=plt.subplot(2,1,2)
            pc=plt.pcolor(self.outputData['time'],r,rws_qc[:,0,:],cmap='coolwarm',vmin=np.nanpercentile(rws_qc,5)-1,vmax=np.nanpercentile(rws_qc,95)+1)
            plt.xlabel('Time (UTC)')
            plt.ylabel(r'Range [m]')
            date_fmt = mdates.DateFormatter('%H:%M:%S')
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.grid()
            cax=fig_rws.add_axes([ax.get_position().x0+ax.get_position().width+0.01,ax.get_position().y0,0.015,ax.get_position().height])
            cbar = plt.colorbar(pc, cax=cax,label='Filtered radial \n'+r' wind speed [m s$^{-1}$]')
            
            
        if self.outputData.attrs['scan_mode']=='PPI':
            fig_angles=plt.figure(figsize=(18,10))
            plt.subplot(2,1,1)
            plt.plot(self.inputData['time'],self.inputData['azimuth'].values.ravel(),'.k',markersize=5,label='Raw azimuth')
            plt.plot(self.azimuth_selected.time,self.azimuth_selected,'.r',markersize=5,label='Selected azimuth')
            plt.plot(self.azimuth_regularized.time,self.azimuth_regularized,'.g',markersize=5,label='Regularized azimuth')
            plt.xlabel('Time (UTC)')
            plt.ylabel(r'Azimuth [$^\circ$]')
            date_fmt = mdates.DateFormatter('%H:%M:%S')
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.legend()
            plt.grid()
            plt.title('Beam angles at '+self.inputData.attrs['location_id']+' on '+datestr(np.nanmean(dt64_to_num(self.inputData['time'])),'%Y-%m-%d')+\
                    '\n File: '+os.path.basename(self.source))
            
            plt.subplot(2,1,2)
            plt.bar(self.azimuth_detected,self.counts,color='k')
            plt.xlabel(r'Detected significant azimuth [$^\circ$]')
            plt.ylabel('Occurrence')
            plt.grid()
         
            
            fig_rws=plt.figure(figsize=(18,6))
            ctr=1
            for i in i_plot:
                time=(self.outputData.time[:,i] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                
                #plot raw rws
                ax=plt.subplot(2,N_plot,ctr)
                pc=plt.pcolor(X,Y,rws[:,:,i],cmap='coolwarm',vmin=np.nanpercentile(rws_qc,5)-1,vmax=np.nanpercentile(rws_qc,95)+1)
                if ctr==1:
                    plt.ylabel(r'$y$ [m]')
                else:
                    ax.set_yticklabels([])
                plt.grid()
            
                xlim=[np.min(self.outputData.x),np.max(self.outputData.x)]
                ylim=[np.min(self.outputData.y),np.max(self.outputData.y)]
                ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.title(datestr(np.nanmin(time),'%H:%M:%S')+' - '+datestr(np.nanmax(time),'%H:%M:%S'))
               
                if ctr==np.ceil(N_plot/2):
                    plt.title('Radial wind speed at '+self.inputData.attrs['location_id']+' on '+datestr(np.nanmean(time),'%Y-%m-%d')+\
                        '\n File: '+os.path.basename(self.source)\
                              +'\n'+ datestr(np.nanmin(time),'%H:%M:%S')+' - '+datestr(np.nanmax(time),'%H:%M:%S'))
            
                if ctr==N_plot:
                    cax=fig_rws.add_axes([ax.get_position().x0+ax.get_position().width+0.01,ax.get_position().y0,0.015,ax.get_position().height])
                    cbar = plt.colorbar(pc, cax=cax,label='Raw radial \n'+r' wind speed [m s$^{-1}$]')
                
                #plot qc'ed rws
                ax=plt.subplot(2,N_plot,ctr+N_plot)
                plt.pcolor(X,Y,rws_qc[:,:,i],cmap='coolwarm',vmin=np.nanpercentile(rws_qc,5)-1,vmax=np.nanpercentile(rws_qc,95)+1)
                plt.xlabel(r'$x$ [m]')
                if ctr==1:
                    plt.ylabel(r'$y$ [m]')
                else:
                    ax.set_yticklabels([])
                xlim=[np.min(self.outputData.x),np.max(self.outputData.x)]
                ylim=[np.min(self.outputData.y),np.max(self.outputData.y)]
                ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.grid()
                
                if ctr==N_plot:
                    if ctr==N_plot:
                        cax=fig_rws.add_axes([ax.get_position().x0+ax.get_position().width+0.01,ax.get_position().y0,0.015,ax.get_position().height])
                        cbar = plt.colorbar(pc, cax=cax,label='Filtered radial \n'+r' wind speed [m s$^{-1}$]')
                ctr+=1
                
        if self.outputData.attrs['scan_mode']=='RHI':
            fig_angles=plt.figure(figsize=(18,10))
            plt.subplot(2,1,1)
            plt.plot(self.inputData['time'],self.inputData['elevation'].values.ravel(),'.k',markersize=5,label='Raw elevation')
            plt.plot(self.elevation_selected.time,self.elevation_selected,'.r',markersize=5,label='Selected elevation')
            plt.plot(self.elevation_regularized.time,self.elevation_regularized,'.g',markersize=5,label='Regularized elevation')
            plt.xlabel('Time (UTC)')
            plt.ylabel(r'Elevation [$^\circ$]')
            date_fmt = mdates.DateFormatter('%H:%M:%S')
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.legend()
            plt.grid()
            plt.title('Beam angles at '+self.inputData.attrs['location_id']+' on '+datestr(np.nanmean(dt64_to_num(self.inputData['time'])),'%Y-%m-%d')+\
                    '\n File: '+os.path.basename(self.source))
            
            
            plt.subplot(2,1,2)
            plt.bar(self.elevation_detected,self.counts,color='k')
            plt.xlabel(r'Detected significant elevation [$^\circ$]')
            plt.ylabel('Occurrence')
            plt.grid()
            
            fig_rws=plt.figure(figsize=(18,6))
            ctr=1
            for i in i_plot:
                time=(self.outputData.time[:,i] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                
                #plot raw rws
                ax=plt.subplot(2,N_plot,ctr)
                pc=plt.pcolor(X,Z,rws[:,:,i],cmap='coolwarm',vmin=np.nanpercentile(rws_qc,5)-1,vmax=np.nanpercentile(rws_qc,95)+1)
                if ctr==1:
                    plt.ylabel(r'$z$ [m]')
                else:
                    ax.set_yticklabels([])
                plt.grid()
            
                xlim=[np.min(self.outputData.x),np.max(self.outputData.x)]
                zlim=[np.min(self.outputData.z),np.max(self.outputData.z)]
                ax.set_box_aspect(np.diff(zlim)/np.diff(xlim))
                plt.xlim(xlim)
                plt.ylim(zlim)
                plt.title(datestr(np.nanmin(time),'%H:%M:%S')+' - '+datestr(np.nanmax(time),'%H:%M:%S'))
               
                if ctr==np.ceil(N_plot/2):
                     plt.title('Radial wind speed at '+self.inputData.attrs['location_id']+' on '+datestr(np.nanmean(time),'%Y-%m-%d')+\
                        '\n File: '+os.path.basename(self.source)\
                              +'\n'+ datestr(np.nanmin(time),'%H:%M:%S')+' - '+datestr(np.nanmax(time),'%H:%M:%S'))

                if ctr==N_plot:
                    cax=fig_rws.add_axes([ax.get_position().x0+ax.get_position().width+0.01,ax.get_position().y0,0.015,ax.get_position().height])
                    cbar = plt.colorbar(pc, cax=cax,label='Raw radial \n'+r' wind speed [m s$^{-1}$]')
                
                #plot qc'ed rws
                ax=plt.subplot(2,N_plot,ctr+N_plot)
                plt.pcolor(X,Z,rws_qc[:,:,i],cmap='coolwarm',vmin=np.nanpercentile(rws_qc,5)-1,vmax=np.nanpercentile(rws_qc,95)+1)
                plt.xlabel(r'$x$ [m]')
                if ctr==1:
                    plt.ylabel(r'$z$ [m]')
                else:
                    ax.set_yticklabels([])
              
                ax.set_box_aspect(np.diff(zlim)/np.diff(xlim))
                plt.xlim(xlim)
                plt.ylim(zlim)
                plt.grid()
                
                if ctr==N_plot:
                    if ctr==N_plot:
                        cax=fig_rws.add_axes([ax.get_position().x0+ax.get_position().width+0.01,ax.get_position().y0,0.015,ax.get_position().height])
                        cbar = plt.colorbar(pc, cax=cax,label='Filtered radial \n'+r' wind speed [m s$^{-1}$]')
                ctr+=1
        
        if self.outputData.attrs['scan_mode']=='3D':
            fig_angles=plt.figure(figsize=(18,10))
            plt.subplot(4,1,1)
            plt.plot(self.inputData['time'],self.inputData['azimuth'].values.ravel(),'.k',markersize=5,label='Raw azimuth')
            plt.plot(self.azimuth_selected.time,self.azimuth_selected,'.r',markersize=5,label='Selected azimuth')
            plt.plot(self.azimuth_regularized.time,self.azimuth_regularized,'.g',markersize=5,label='Regularized azimuth')
            plt.ylabel(r'Azimuth [$^\circ$]')
            date_fmt = mdates.DateFormatter('%H:%M:%S')
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.legend()
            plt.grid()
            plt.title('Beam angles at '+self.inputData.attrs['location_id']+' on '+datestr(np.nanmean(dt64_to_num(self.inputData['time'])),'%Y-%m-%d')+\
                    '\n File: '+os.path.basename(self.source))
            
                    
            plt.subplot(4,1,2)
            plt.plot(self.inputData['time'],self.inputData['elevation'].values.ravel(),'.k',markersize=5,label='Raw elevation')
            plt.plot(self.elevation_selected.time,self.elevation_selected,'.r',markersize=5,label='Selected elevation')
            plt.plot(self.elevation_regularized.time,self.elevation_regularized,'.g',markersize=5,label='Regularized elevation')
            plt.xlabel('Time (UTC)')
            plt.ylabel(r'Elevation [$^\circ$]')
            date_fmt = mdates.DateFormatter('%H:%M:%S')
            plt.gca().xaxis.set_major_formatter(date_fmt)
            plt.legend()
            plt.grid()
            
            ax=plt.subplot(2,1,2)
            plt.plot(self.azimuth_detected,self.elevation_detected,'--k')
            sc=plt.scatter(self.azimuth_detected,self.elevation_detected,c=self.counts,cmap='hot')
            plt.xlabel(r'Azimuth [$^\circ$]')
            plt.ylabel(r'Elevation [$^\circ$]')
            plt.grid()
            cax=fig_angles.add_axes([ax.get_position().x0+ax.get_position().width+0.01,ax.get_position().y0,0.015,ax.get_position().height])
            cbar = plt.colorbar(sc, cax=cax,label='Occurrence')
            ax.set_facecolor('lightgrey')
            
            fig_rws=plt.figure(figsize=(18,9))
            ctr=1
            for i in i_plot:
                time=(self.outputData.time[:,i] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                x=self.outputData.x[:,:,i].values
                y=self.outputData.y[:,:,i].values
                z=self.outputData.z[:,:,i].values
                f=rws[:,:,i].values
                real=~np.isnan(x+y+z+f)
                
                if np.sum(real)>10000:
                    skip=int(np.sum(real)/10000)
                else:
                    skip=1
                
                #plot raw rws
                xlim=[np.min(self.outputData.x),np.max(self.outputData.x)]
                ylim=[np.min(self.outputData.y),np.max(self.outputData.y)]
                zlim=[np.min(self.outputData.z),np.max(self.outputData.z)]
                
                ax=plt.subplot(2,N_plot,ctr, projection='3d')
                sc=ax.scatter(x[real][::skip],y[real][::skip],z[real][::skip],s=2,c=f[real][::skip],cmap='coolwarm',vmin=np.nanpercentile(rws_qc,5)-1,vmax=np.nanpercentile(rws_qc,95)+1)
                if ctr==1:
                    ax.set_xlabel(r'$x$ [m]',labelpad=10)
                    ax.set_ylabel(r'$y$ [m]')
                    ax.set_zlabel(r'$z$ [m]')
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])
                
                ax.set_box_aspect((np.diff(xlim)[0],np.diff(ylim)[0],np.diff(zlim)[0]))
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
                plt.title(datestr(np.nanmin(time),'%H:%M:%S')+' - '+datestr(np.nanmax(time),'%H:%M:%S'))
               
                if ctr==np.ceil(N_plot/2):
                    plt.title('Radial wind speed at '+self.inputData.attrs['location_id']+' on '+datestr(np.nanmean(time),'%Y-%m-%d')+\
                        '\n File: '+os.path.basename(self.source)\
                              +'\n'+ datestr(np.nanmin(time),'%H:%M:%S')+' - '+datestr(np.nanmax(time),'%H:%M:%S'))
            
                if ctr==N_plot:
                    cax=fig_rws.add_axes([ax.get_position().x0+ax.get_position().width+0.035,ax.get_position().y0,0.015,ax.get_position().height])
                    cbar = plt.colorbar(sc, cax=cax,label='Raw radial \n'+r' wind speed [m s$^{-1}$]')
                
                #plot qc'ed rws
                f=rws_qc[:,:,i].values
                real=~np.isnan(x+y+z+f)
                ax=plt.subplot(2,N_plot,ctr+N_plot, projection='3d')
                sc=ax.scatter(x[real],y[real],z[real],s=2,c=f[real],cmap='coolwarm',vmin=np.nanpercentile(rws_qc,5)-1,vmax=np.nanpercentile(rws_qc,95)+1)
                
                if ctr==1:
                    ax.set_xlabel(r'$x$ [m]',labelpad=10)
                    ax.set_ylabel(r'$y$ [m]')
                    ax.set_zlabel(r'$z$ [m]')
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])
    
                ax.set_box_aspect((np.diff(xlim)[0],np.diff(ylim)[0],np.diff(zlim)[0]))
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
                plt.grid()
                
                if ctr==N_plot:
                    if ctr==N_plot:
                        cax=fig_rws.add_axes([ax.get_position().x0+ax.get_position().width+0.035,ax.get_position().y0,0.015,ax.get_position().height])
                        cbar = plt.colorbar(sc, cax=cax,label='Filtered radial \n'+r' wind speed [m s$^{-1}$]')
                ctr+=1
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.25)
        if not self.outputData.attrs['scan_mode']=='Stare':
            fig_angles.savefig(self.save_filename.replace('.nc','_angles.png'))
        fig_probability.savefig(self.save_filename.replace('.nc','_probability.png'))
        fig_rws.savefig(self.save_filename.replace('.nc','_wind_speed.png'))
        
        self.print_and_log(f'QC figures saved in {os.path.dirname(self.save_filename)}')
                
def datestr(num,format='%Y-%m-%d %H:%M:%S.%f'):
    '''
    Unix time to string of custom format
    '''
    from datetime import datetime
    string=datetime.utcfromtimestamp(num).strftime(format)
    return string

def dt64_to_num(dt64):
    '''
    numpy.datetime64[ns] time to Unix time
    '''
    tnum=(dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return tnum
        
def floor(value, step):
    '''
    A helper method for floor operation
    '''
    return np.floor(value / step) * step

        
def ceil(value, step):
    '''
    A helper method for ceil operation
    '''
    return np.ceil(value / step) * step

def local_probability(df, config):
    
    '''
    Calculates 2D historgam of normalized RWS and SNR
    
    Inputs:
    -----
    df: dataframe
        dataframe of data and current QC flag
    
    config: dict
        configuration
        
    Outputs:
    -----
    filt: dataframe
        QC flag
    df: dataframe
        data
    rws_range: series
        normalized radial wind speed range vs. probability
    probability_threshold: float
        probability lower threshold
    '''
    
    #Set up bin sizes
    df['rws_filt']=df['rws_norm'].where(df['filtered_temp'])
    df['snr_filt']=df['snr_norm'].where(df['filtered_temp'])
    delta_rws = 3.49 * df['rws_filt'].std() / np.sum(~np.isnan(df['rws_filt'].unique())) ** (1 / 3)
    eps = 10**-10
    
    rws_bins = np.arange(
        np.floor(df['rws_filt'].min() / delta_rws) * delta_rws,
        np.ceil(df['rws_filt'].max() / delta_rws) * delta_rws + eps,
        delta_rws,
    )
    
    delta_snr = 3.49 * df['snr_filt'].std() / np.sum(~np.isnan(df['snr_filt'].unique())) ** (1 / 3)
    snr_bins = np.arange(
        np.floor(df['snr_filt'].min() / delta_snr) * delta_snr,
        np.ceil(df['snr_filt'].max() / delta_snr) * delta_snr + eps,
        delta_snr,
    )
    
    #Calculate 2-D pdf
    df['rws_bins'] = pd.cut(df['rws_filt'], rws_bins)
    df['snr_bins'] = pd.cut(df['snr_filt'], snr_bins)
    groups = df.groupby(['rws_bins', 'snr_bins'])
    count=groups['rws_filt'].count()
    
    temp = df.set_index(['rws_bins', 'snr_bins']).copy()
    temp['probability'] = count/count.max()
    df['probability'] = temp['probability'].reset_index(drop=True)
    
    #Identify probability threshold that excludes data with large scattering
    probability_bins=np.linspace(np.log10(df['probability'].min()+eps)-1,np.log10(df['probability'].max()),config['N_probability_bins'])
    df['probability_bins'] = pd.cut(np.log10(df['probability']), probability_bins)
    groups = df.groupby(['probability_bins'])
    rws_range=groups['rws_norm'].apply(lambda x: np.nanpercentile(x,config['max_percentile'])-np.nanpercentile(x,config['min_percentile']))
    max_rws_range=rws_range.min()+config['rws_norm_increase_limit']*(rws_range.max()-rws_range.min())
    if np.max(rws_range)>max_rws_range:
        i_threshold=np.where(rws_range>max_rws_range)[0][-1]+1
        probability_threshold=10**rws_range.index[i_threshold].left
    else:
        probability_threshold=config['min_probability_range']
        
    if probability_threshold>config['max_probability_range']:
        probability_threshold=config['max_probability_range']
        
    filt= df['probability'] >probability_threshold

    return filt,df,rws_range,probability_threshold

def lidar_xyz(r, ele, azi):
    '''
    Convert spherical to Cartesian coordinates
    '''
    R = r 
    A = np.pi/2-np.radians(azi)  
    E = np.radians(ele) 

    X = R * np.cos(E) * np.cos(A)
    Y = R * np.cos(E) * np.sin(A)
    Z = R * np.sin(E)

    return X, Y, Z


def defineLocalBins(df, config):
    '''
    Helper function for making tidy bins based on ranges and bin sizes
    '''
    def bins(coord, delta):
        return np.arange(
            np.floor(coord.min())-delta/2,
            np.ceil(coord.max())+delta,
            delta,
        )

    if 'dx' in config.keys():
        df['xbins'] = pd.cut(df.x, bins(df.x, config['dx']))
    if 'dy' in config.keys():
        df['ybins'] = pd.cut(df.y, bins(df.y, config['dy']))
    if 'dz' in config.keys():
        df['zbins'] = pd.cut(df.z, bins(df.z, config['dz']))
    if 'dtime' in config.keys():
        df['timebins'] = pd.cut(df.deltaTime, bins(df.deltaTime, config['dtime']))

    return df

def mid(x):
    '''
    Mid point in vector
    '''
    return (x[:-1]+x[1:])/2

def gaussian(x,sigma):
    '''
    Gaussian function
    '''
    return np.exp(-x**2/(2*sigma**2))

if __name__ == '__main__':
    '''
    Test block
    '''
    cd=os.path.dirname(__file__)
        
    source='data/360ppi-csm/rt5.lidar.z02.a0.20230711.173203.user1.nc'
    config_file='config/configs_standardize.xlsx'

    # Create an instance of LIDARGO
    lproc = LIDARGO(source,config_file, verbose=True)
    
    # Run processing
    lproc.process_scan(make_figures=True, replace=True,save_file=True)