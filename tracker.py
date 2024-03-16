import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import metpy.calc as mpcalc
from datetime import datetime
from metpy.units import units
from cartopy.geodesic import Geodesic
from netCDF4 import Dataset

#define a class for TC
class TC:
    def __init__(self,number,name,index):
        self.number = number#台风编号
        self.name = name
        self.point = 0 #the number of records in the best track data
        self.track = None #two dimension array, every row is a record of : time, latitude, longitude, level,pressure, wind speed 
        self.angle = None
        self.index = index#数据库文件名
        self.sudden_change = None

def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371 # Radius of Earth in kilometers. Use 3956 for miles
        return c * r

def get_max_vor(upper,lon_min_mslp, lat_min_mslp): # get the location of maximum relative vorticity in 850 hPa within a radius of 278 km around the MSLP minimum
    # get the maxima in  850 hPa relative vorticity  within a radius of 278 km around the MSLP minimum
     # Extract latitude, longitude, and vorticity from the dataset
    lat = upper.variables['latitude'][:]
    lon = upper.variables['longitude'][:]
    u850 = upper.variables['u'][0,2,:,:]
    v850 = upper.variables['v'][0,2,:,:]
    u850 = units.Quantity(u850, 'm/s')
    v850 = units.Quantity(v850, 'm/s')
    dx, dy = mpcalc.lat_lon_grid_deltas(lon,lat)
    vor = mpcalc.vorticity(u850, v850, dx=dx, dy=dy)

    # Generate a 2D grid of latitudes and longitudes
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Calculate distances using the Haversine formula (re-use the function from earlier)
    distances = haversine(lon_min_mslp, lat_min_mslp, lon_grid, lat_grid)

    # Find grid points within a 278 km radius
    within_radius = distances <= 278

    # Filter the vorticity values based on the within_radius mask
    vor_within_radius = vor[within_radius]
    max_vor_value = np.max(vor_within_radius)
    # make sure vor_value is a scalar
    max_vor_value = max_vor_value.magnitude
    # Find the index of the maximum value within the filtered array
    max_vor_index_within = np.argmax(vor_within_radius)
    # Convert this index back to the original lat/lon grid indexes
    original_indexes = np.where(within_radius)
    max_lat_index = original_indexes[0][max_vor_index_within]
    max_lon_index = original_indexes[1][max_vor_index_within]
    lon_max_vor = lon[max_lon_index]
    lat_max_vor = lat[max_lat_index]
    return lon_max_vor, lat_max_vor, max_vor_value

def ax_add_feature(ax,extent=[100, 150, 0, 40]):
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, edgecolor='black')
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.3, edgecolor='grey')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='-', linewidth=0.3, edgecolor='black')
    ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.5, facecolor='skyblue', edgecolor='black')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.3, facecolor='lightblue')
    ax.set_extent([100, 150, 0, 40])
    return ax

def find_location_min_mslp(upper,surface,last_track):

    # get the corresoonding best track data of lat and lon
    lat_last = last_track[-1][0]
    lon_last = last_track[-1][1]
    # find the minimum mslp in surface data within a radius of 445 km around the last track
    lat = surface.variables['latitude'][:]
    lon = surface.variables['longitude'][:]
    msl = surface.variables['msl'][0,:,:]

    # Create a 2D grid of lat and lon values
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Calculate distances from each grid point to the specified point
    distances = haversine(lon_last, lat_last, lon_grid, lat_grid)

    # Find the indices where the distance is within 445 km
    within_radius = distances <= 445
    msl_within_radius = msl[within_radius]

    # Filter the msl values based on the within_radius mask and find the minimum MSLP
    min_mslp_value = np.min(msl[within_radius])

    # Find the index of the minimum value within the filtered array
    min_mslp_index_within = np.argmin(msl_within_radius)

    # Convert this index back to the original lat/lon grid indexes
    original_indexes = np.where(within_radius)
    min_lat_index = original_indexes[0][min_mslp_index_within]
    min_lon_index = original_indexes[1][min_mslp_index_within]
    lon_min_mslp = lon[min_lon_index]
    lat_min_mslp = lat[min_lat_index]

    # get the maxima in  850 hPa relative vorticity  within a radius of 278 km around the MSLP minimum.
    lon_max_vor, lat_max_vor,max_vor_value = get_max_vor(upper, lon_min_mslp, lat_min_mslp)

    # Create a map
    #fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    #fig.set_size_inches(30, 30)
    #ax = ax_add_feature(ax)

    # plot the location of the last track  and mark the 445 km radius, plot the location of the minimum mslp, plot the location of the maximum 850 hPa relative vorticity and mark the 278 km radius
    #ax.plot(lon_last, lat_last, marker='o', color='black', markersize=10, transform=ccrs.Geodetic())
    #ax.plot(lon_min_mslp, lat_min_mslp, marker='o', color='blue', markersize=10, transform=ccrs.Geodetic())
    #geodetic = Geodesic()
    #circle = geodetic.circle(lon=lon_last, lat=lat_last, radius=445000, n_samples=100, endpoint=False)
    #ax.plot(circle[:, 0], circle[:, 1], color='black', transform=ccrs.Geodetic(), linestyle='--')
    #circle2 = geodetic.circle(lon=lon_min_mslp, lat=lat_min_mslp, radius=278000, n_samples=100, endpoint=False)
    #ax.plot(circle2[:, 0], circle2[:, 1], color='blue', transform=ccrs.Geodetic(), linestyle='--')
    #ax.plot(lon_max_vor, lat_max_vor, marker='o', color='orange', markersize=10, transform=ccrs.Geodetic())
    #plt.show()
    return lon_min_mslp, lat_min_mslp,min_mslp_value,  max_vor_value, lon_max_vor, lat_max_vor

def main():
    # Load the data from './data/sudden_change_all.xlsx' as ground truth
    data = pd.read_excel('./data/sudden_change_all.xlsx')
    time_start = 2021091200
    time_end = 2021091812

    # Filter the data from time_start to time_end
    data = data[(data['时间'] >= time_start) & (data['时间'] <= time_end)]

    #load the best track data
    number = int(data['台风编号'].values[0])
    name = data['台风名称'].values[0]
    index = data['数据库文件名'].values[0]

    #apeend the track
    track = None
    angle = None
    sudden_change = []
    for record in data.values:
        if angle is None:
            angle = np.array([record[9]])
        else:
            angle = np.append(angle,[record[9]],axis=0)
            if record[9]>45: # sudden change, stored in a index
                sudden_change.append(len(angle)-1)
        if track is None:
            track = np.array([record[3:9]])
        else:
            track = np.append(track,[record[3:9]],axis=0)
    tc_ground_truth = TC(number,name,index)
    tc_ground_truth.point = len(data)
    tc_ground_truth.track = track
    tc_ground_truth.angle = angle
    tc_ground_truth.sudden_change = sudden_change

    '''
    create a temporary array to store last output of find_location_min_mslp
    the first row is the first row in ground truth, as first time we use the obersevation data to find the location of MSLP minimum
    then observation is not available, so we use the last output of the forecast to find the next location of MSLP minimum
    '''
    last_track = np.array([[tc_ground_truth.track[0][1],tc_ground_truth.track[0][2]]])
    #lets say we do a 3 days forecast, every 6 hours, so we have 13 reanalysis nc data. each nc data is combined with a upper layer and a surface layer, which are located in './data/upper' and './data/surface' respectively
    #the first data name is '0_upper.nc', '0_surface.nc', the second data name is '6_upper.nc', '6_surface.nc', and so on, until the last data name is '72_upper.nc', '72_surface.nc'
    tc_forecast = TC(number,name,index)
    track = None
    for i in range(0,13):
        upper = Dataset('./data/upper/'+str(i*6)+'_upper.nc')
        surface = Dataset('./data/surface/'+str(i*6)+'_surface.nc')
        #find the location of the minimum mslp
        lon,lat,pressure,vorticity = find_location_min_mslp(upper,surface,last_track)[0:4]
        #append the track to last_track
        last_track = np.append(last_track,[[lat,lon]],axis=0)
        new_track = np.array([0,lat,lon,pressure,vorticity,0])
        if track is None:
            track = [new_track]
        else:
            track = np.append(track,[new_track],axis=0)
    tc_forecast.track = track

    #plot the track of both ground truth and forecast
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    fig.set_size_inches(30, 30)
    ax = ax_add_feature(ax)

    ax.plot(tc_ground_truth.track[:,2],tc_ground_truth.track[:,1],marker = 'o',markersize = 3,linewidth=1,transform=ccrs.PlateCarree())
    ax.plot(tc_forecast.track[:,2],tc_forecast.track[:,1],marker = 'o',markersize = 3,linewidth=1,transform=ccrs.PlateCarree())
    ax.scatter(tc_ground_truth.track[tc_ground_truth.sudden_change,2],tc_ground_truth.track[tc_ground_truth.sudden_change,1],color='r',marker = 'o',s=30)
    plt.show()

    #store both the ground truth and forecast in one xlsx file from the starting time to the end time, and store the file in './data/output.xlsx'
    length = max(tc_ground_truth.track.shape[0], tc_forecast.track.shape[0])
    forecast_padded = np.full((length, tc_forecast.track.shape[1]), np.nan)
    forecast_padded[:tc_forecast.track.shape[0], :tc_forecast.track.shape[1]] = tc_forecast.track
    tc_forecast.track = forecast_padded
    tc_forecast.track

    df = pd.DataFrame({
        'time': tc_ground_truth.track[:,0],
        'lat_ground_truth' : tc_ground_truth.track[:,1],
        'lon_ground_truth' : tc_ground_truth.track[:,2],
        'level_ground_truth' : tc_ground_truth.track[:,3],
        'pressure_ground_truth' : tc_ground_truth.track[:,4],
        'wind_speed_ground_truth' : tc_ground_truth.track[:,5],
        'lat_forecast' : tc_forecast.track[:,1],
        'lon_forecast' : tc_forecast.track[:,2],
        'pressure_forecast' : tc_forecast.track[:,3],
        'vorticity_forecast' : tc_forecast.track[:,4],
    }
    )

    df.replace(np.nan, '', inplace=True)
    df.to_excel('./data/output.xlsx', index=False)

if __name__ == "__main__":
    main()