'''
Convert hpl Halo files to netCDF in WDH a0 format
'''
import os
cd=os.path.dirname(__file__)

import xarray as xr
import numpy as np
import shutil
import re
from matplotlib import pyplot as plt

class LIDARGO_format():
    def __init__(self,verbose=True,logger=None):
        '''
        Initialize the LIDAR data formatter
        
        Inputs:
        ------
        logger: logger object
            external error logger (if any)
        level_out: str
            output data level
       
        '''
        plt.close('all')
        
        self.logger=logger
        self.verbose=verbose
        
        
    def process_scan(self, source, model, site,z_id, level_out, save_file=True, save_path=None, replace=True):
        
        '''
        Format the raw scan file into WDH-compatible netCDF file
        
        Inputs:
        ------
        source: str
            The input-level file name 
        model: str
            model of the lidar
        site: str
            site short name
        z_id: str
            ID of the lidar if multiple are deployed at same site (01,02, etc)
        verbose: bool
            Whether or not print QC-related information
        save_file: bool
            Whether or not save the final processed file
        save_path: str
            String with the directory the destination of the processed files.
            Optional, defaults to analogous input, replacing input-level with output-level
            data. Creates the necessary intermediate directories
        replace: bool
            Whether or not to replace processed scan if one already exists
        
        '''
        
        source00=self.rename(source,model,site,z_id,save_path,replace)
        
        self.read(source00,model)
        

    def rename(self,source,model,site,z_id,save_path=None,replace=False):
        if save_path is None:
            save_path=os.path.join('/'.join(os.path.dirname(source).split('/')[:-1]),site+'.lidar.'+z_id+'.'+'00')
        os.makedirs(save_path,exist_ok=True)
        
        if model=='Halo XR':
            if 'Stare' in source:
                pattern = r"Stare_\d+_(\d{8})_(\d{2})_(.*?)\.hpl"
                scan_type='stare'
            elif 'User' in source:
                pattern = r"User\d{1}_\d+_(\d{8})_(\d{6})\.hpl"
                scan_type='user'+source[source.find('User')+4]
            else:
                
                raise NameError(f"Source of {source} not supported.")
                
            match = re.search(pattern, os.path.basename(source))
            date_part = match.group(1)  
            if len(match.group(2))<6:
                with open(source, "r") as f:
                    lines = []
                    for line_num in range(11):
                        lines.append(f.readline())
                    metadata={}
                    for line in lines:
                        metaline = line.split(":")
                        if "Start time" in metaline:
                            metadata["Start time"] = metaline[1:]
                        else:
                            metadata[metaline[0]] = metaline[1]  # type: ignore
                    time_part = match.group(2)+metadata["Start time"] [1]+metadata["Start time"] [2].split('.')[0]
            else:
                time_part = match.group(2)
                
            save_filename=site+'.lidar.'+z_id+'.'+'00'+'.'+date_part+'.'+time_part+'.'+scan_type+os.path.splitext(source)[1]
            if os.path.exists(os.path.join(save_path,save_filename))==False or replace==True:
                shutil.copyfile(source, os.path.join(save_path,save_filename))
        return os.path.join(os.path.join(save_path,save_filename))

    def read(self,source,model):
        
        if model=='Halo XR':
            with open(source, "r") as f:
                lines = []
                for line_num in range(11):
                    lines.append(f.readline())
        
                # read metadata into strings
                metadata = {}
                for line in lines:
                    metaline = line.split(":")
                    if "Start time" in metaline:
                        metadata["Start time"] = metaline[1:]
                    else:
                        metadata[metaline[0]] = metaline[1]  # type: ignore
        
                # convert some metadata
                num_gates = int(metadata["Number of gates"])  # type: ignore
        
                # Read some of the label lines
                for line_num in range(6):
                    f.readline()
        
                # initialize arrays
                time = []
                azimuth = []
                elevation = []
                pitch = []
                roll = []
                doppler = []
                intensity = []
                beta = []
        
                while True:
                    a = f.readline().split()
                    if not len(a):  # is empty
                        break
        
                    time.append(float(a[0]))
                    azimuth.append(float(a[1]))
                    elevation.append(float(a[2]))
                    pitch.append(float(a[3]))
                    roll.append(float(a[4]))
        
                    doppler.append([0] * num_gates)
                    intensity.append([0] * num_gates)
                    beta.append([0] * num_gates)
        
                    for _ in range(num_gates):
                        b = f.readline().split()
                        range_gate = int(b[0])
                        doppler[-1:][0][range_gate] = float(b[1])
                        intensity[-1:][0][range_gate] = float(b[2])
                        beta[-1:][0][range_gate] = float(b[3])
        
            # convert date to np.datetime64
            start_time_string = "{}-{}-{}T{}:{}:{}".format(
                metadata["Start time"][0][1:5],  # year
                metadata["Start time"][0][5:7],  # month
                metadata["Start time"][0][7:9],  # day
                "00",  # hour
                "00",  # minute
                "00.00",  # second
            )
        
            # find times where it wraps from 24 -> 0, add 24 to all indices after
            new_day_indices = np.where(np.diff(time) < -23)
            for new_day_index in new_day_indices[0]:
                time[new_day_index + 1 :] += 24.0
        
            start_time = np.datetime64(start_time_string)
            datetimes = [
                start_time + np.timedelta64(int(3600 * 1e6 * dtime), "us") for dtime in time
            ]
        
            outputData = xr.Dataset(
                {
                    # "time": (("time"), time),
                    "azimuth": (("time"), azimuth),
                    "elevation": (("time"), elevation),
                    "pitch": (("time"), pitch),
                    "roll": (("time"), roll),
                    "wind_speed": (("time", "range_gate"), doppler),
                    "intensity": (("time", "range_gate"), intensity),
                    "beta": (("time", "range_gate"), beta),
                },
                coords={"time": np.array(datetimes), "range_gate": np.arange(num_gates)},
                attrs={"Range gate length (m)": float(metadata["Range gate length (m)"])},  # type: ignore
            )
        
            # Save some attributes
            outputData.attrs["Range gate length (m)"] = float(
                metadata["Range gate length (m)"]  # type: ignore
            )
            outputData.attrs["Number of gates"] = float(metadata["Number of gates"])  # type: ignore
            outputData.attrs["Scan type"] = str(metadata["Scan type"]).strip()
            outputData.attrs["Pulses per ray"] = float(metadata["Pulses/ray"])  # type: ignore
            outputData.attrs["System ID"] = int(metadata["System ID"])  # type: ignore
            outputData.attrs["source"] = str(metadata["source"])[1:-5]
            outputData.attrs["code_version"]=''
            outputData.attrs["title"]='Lidar Halo XRP'
            outputData.attrs["description"]='AWAKEN XRP Halo Lidar data'
            outputData.attrs["location_id"]=os.path.basename(source).split('.')[0]
            
            outputData["distance"] = (
                "range_gate",
                outputData.coords["range_gate"].data * outputData.attrs["Range gate length (m)"]
                + outputData.attrs["Range gate length (m)"] / 2,
            )
            outputData["distance_overlapped"] = (
                "range_gate",
                outputData.coords["range_gate"].data * 1.5
                + outputData.attrs["Range gate length (m)"] / 2,
            )
            intensity = outputData.intensity.data.copy()
            intensity[intensity <= 1] = np.nan
            outputData['SNR']=xr.DataArray(data=10 * np.log10(intensity - 1),coords={"time": np.array(datetimes), "range_gate": np.arange(num_gates)})
        
            # Dynamically add scan type and z-id (z02, z03, etc) to outputData metadata
            # loc_id, instrument, z02/z03, data '00', date, time, scan type, extension
            raw_basename = source.replace("\\", "/").rsplit("/")[-1]
            if ".z" in raw_basename:
                _, _, z_id, _, _, _, scan_type, _ = raw_basename.lower().split(".")
            else:  # local NREL tsdat-ing
                z_id = str(outputData.attrs["System ID"])
                scan_type = ""
                if "user" in raw_basename.lower():
                    scan_type = raw_basename.split("_")[0].lower()
                elif "stare" in raw_basename.lower():
                    scan_type = "stare"
                elif "vad" in raw_basename.lower():
                    scan_type = "vad"
                elif "wind_profile" in raw_basename.lower():
                    scan_type = "wind_profile"
                elif "rhi" in raw_basename.lower():
                    scan_type = "rhi"
        
            valid_types = ["user", "stare", "vad", "wind_profile", "rhi"]
            if not any(valid_type in scan_type for valid_type in valid_types):
                raise NameError(f"Scan type '{scan_type}' not supported.")
        
            outputData.attrs["scan_type"] = scan_type
            outputData.attrs["z_id"] = z_id
    
        return outputData

    def print_and_log(self,message):
        if self.verbose:
            print(message)
        if self.logger is not None:
            self.logger.info(message)

if __name__ == '__main__':
    '''
    Test block
    '''
    source='C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Technical/LiDAR/LIDARGO/data/Example1/User4_137_20230830_064606.hpl'
    site='sc1'
    model='Halo XR'
    z_id='01'
    level_out='a0'
    
    lproc=LIDARGO_format()
    lproc.process_scan(source,model, site,z_id, level_out, replace=False,save_file=True)
    
    
    