# %%
import pandas as pd
import numpy as np
import xarray as xr
import os
import warnings
warnings.filterwarnings("ignore")

def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371.0  
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def locate(last_lon,last_lat,t,index):  #given the last location, find the new location at t
    file_name = file_path + str(6*(index+1)) + '_surface_' + str(t) + '.nc'
    ds = xr.open_dataset(file_name).sel(latitude = slice(70,-20),longitude = slice(90,180))
    lons = ds['longitude'].values  
    lats = ds['latitude'].values   
    msl = ds['msl'].values[0]      

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    distances = haversine_distance(last_lon, last_lat, lon_grid, lat_grid)
    mask = distances <= 445

    min_msl_index = np.argmin(msl[mask]) 
    min_msl_value = round(msl[mask][min_msl_index]/100,1)
    min_lat = lat_grid[mask][min_msl_index]
    min_lon = lon_grid[mask][min_msl_index]
    
    return min_lon, min_lat, min_msl_value

#%%
number = 'your TC number'
file_path = 'your path'
files = os.listdir(file_path)
file_nc = [f for f in files if ('output' in f and '.nc' in f) ]
file_nc.sort(key = lambda x: (int(x.split('_')[1]),x.split('_')[2]))
time_start = int(file_nc[0].split('_')[-1].split('.')[0])
time_end = int(file_nc[-1].split('_')[-1].split('.')[0])

sheet_name = 0 #sudden change : 0 | normal : 1
data = pd.read_excel('/HOME/scw6c93/run/pangu/tracker/track_both.xlsx',sheet_name=sheet_name)
best_track = data[(data['台风编号'] == number) & (data['时间'] <= time_end) & (data['时间'] >= time_start) ]
time = best_track['时间'].values
timef = pd.to_datetime(time,format = '%Y%m%d%H')
first_row_index = best_track.index.min()
previous_row = data.loc[first_row_index - 1]

era5 = np.zeros((len(time), 3)) # | lon | lat | pressure 
era5[:,0] = best_track['经度']
era5[:,1] = best_track['纬度']
era5[:,2] = best_track['气压']

last_lon = previous_row['经度']
last_lat = previous_row['纬度']
pangu = np.zeros((len(time), 3))    # | lon | lat | pressure 

for index,t in enumerate(time):
    pangu[index][0], pangu[index][1], pangu[index][2] = locate(last_lon,last_lat,t,index)
    last_lon, last_lat = pangu[index][0], pangu[index][1]

# %% save everthing
columns = ['lon','lat', 'pressure']
df1 = pd.DataFrame(pangu, columns=columns)
df2 = pd.DataFrame(era5,columns=columns)
time_series  = pd.Series(timef,name = 'time')
df1.insert(0,'time',time_series)
df2.insert(0,'time',time_series)

output_file = file_path + '/track.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df1.to_excel(writer, sheet_name='pangu', index=False)
    df2.to_excel(writer, sheet_name='era5', index=False)

