#= ω =  /꧂
#让脚本不出bug的魔法
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import metpy.calc as mpcalc
import datetime
from metpy.units import units
from cartopy.geodesic import Geodesic
from netCDF4 import Dataset
import plotly.graph_objs as go

#define a class for TC
class TC:
    def __init__(self,number,name,index):
        self.number = number#台风编号
        self.name = name
        self.point = 0 
        self.track = None #two dimension array, every row is a record of : time, latitude, longitude, level,pressure, wind speed 
        self.angle = None
        self.index = index#数据库文件名
        self.sudden_change = None
        self.false_forecast = None #相对涡度低于5*10^-5/s的预报的indices
        self.type = None # 0 为 best track, 1 为 forecast

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
    return lon_min_mslp, lat_min_mslp,min_mslp_value,  max_vor_value, lon_max_vor, lat_max_vor
        

def get_xlsx(tc_ground_truth,tc_forecast_list): #将best track和所有forecast的track写入xlsx文件
    writer = pd.ExcelWriter('./data/output/track.xlsx')
    df = pd.DataFrame(tc_ground_truth.track,columns=['time','lat','lon','level','pressure','wind speed'])
    df.to_excel(writer,sheet_name='best track',index=False)
    for i in range(len(tc_forecast_list)):
        df = pd.DataFrame(tc_forecast_list[i].track,columns=['time','lat','lon','pressure','vorticity',' '])
        df.to_excel(writer,sheet_name='forecast'+str(i+1),index=False)
    writer.close()

# 根据台风强度定义颜色
def get_color_based_on_intensity(level, angle):
    if angle > 45:
        return 'red'
    else:
        # 可以根据需要调整颜色映射
        intensity_color_map = {
            0: 'white',
            1: 'lightblue',
            2: 'blue',
            3: 'green',
            4: 'pink',
            5: 'yellow',
            6: 'orange',
            9: 'gray'
        }
        return intensity_color_map.get(level, 'gray')  # 如果强度不在0到9之间，使用灰色
def get_color_for_forecast(i):
    j = i%7
    color_map = ['aquamarine','Antiquewhite','Chartreuse','Coral','Lavenderblush','Lightyellow','Pink']
    return color_map[j]

def generate_hover_text(tc_track, tc_forecast_list):
    hover_texts = {tuple(point[1:3]):[f"路径: best track <br>时间: {point[0]}<br>经度: {point[2]}<br>纬度: {point[1]}<br>等级：{point[3]}<br>气压：{point[4]}<br>风速：{point[5]}<br>角度：{angle}<br>"] for point ,angle in zip(tc_track.track, tc_track.angle)}
    for i,forecast in enumerate(tc_forecast_list):
        for point in forecast.track:
            if tuple(point[1:3]) in hover_texts:
                hover_texts[tuple(point[1:3])].append(f"路径: forecast{i+1}<br>时间: {point[0]}<br>经度: {point[2]}<br>纬度: {point[1]}<br>气压：{point[3]}<br>相对涡度：{point[4]}<br>")
            else:
                hover_texts[tuple(point[1:3])] = [f"路径: forecast{i+1}<br>时间: {point[0]}<br>经度: {point[2]}<br>纬度: {point[1]}<br>气压：{point[3]}<br>相对涡度：{point[4]}<br>"]
    for key in hover_texts.keys():
        hover_texts[key] = '<br>'.join(hover_texts[key])
    return hover_texts

def plot_track(tc,tc_forecast_list,hover_texts): #绘制best track和所有forecast的track
    mapbox_access_token = "pk.eyJ1IjoiMTE3MzQ3MTE1NCIsImEiOiJjbHJ5ZjBneG0wc3BuMmxwOGh5bnp3ZHcwIn0.2IelRzFH5GXB8sCudQDSrw"
    # 绘制台风路径
    trace = go.Scattermapbox(
        lon = tc.track[:, 2],  # 经度
        lat = tc.track[:, 1],  # 纬度
        line = dict(
            color = 'rgba(0, 0, 139, 0.5)' , # 深蓝色，透明度为0.5
            width = 3
        ),
        mode = 'lines+markers',
        name = 'best track',
        marker=go.scattermapbox.Marker(
        size = 5,
        color = [get_color_based_on_intensity(point[3], angle) for point, angle in zip(tc.track, tc.angle)],
    ),
    text = [hover_texts[tuple(point[1:3])] for point in tc.track],

)
    
    # 绘制预报路径
    traces = []
    for i in range(len(tc_forecast_list)):
        trace_forecast = go.Scattermapbox(
            lon = tc_forecast_list[i].track[:, 2],  # 经度
            lat = tc_forecast_list[i].track[:, 1],  # 纬度
            line = dict(
                color = get_color_for_forecast(i),
                width = 3
            ),
            mode = 'lines+markers',
            name = 'forecast'+str(i+1),
            marker=go.scattermapbox.Marker(
            size = 5,
            color = ['red' if point[4] < 5*10**-5 else 'pink' for point in tc_forecast_list[i].track],
        ),
        text = [hover_texts[tuple(point[1:3])] for point in tc_forecast_list[i].track],
    )
        traces.append(trace_forecast)

    # 设置地图样式
    layout = go.Layout(
        title = str(tc.number)+' '+str(tc.name),
        mapbox = go.layout.Mapbox(
            accesstoken = mapbox_access_token,
            bearing = 0,
            center = go.layout.mapbox.Center(lat = 20, lon = 130),
            pitch = 0,
            zoom = 3,
            style = 'mapbox://styles/1173471154/clrzqawsk00a201po7kev1jdk',  # 或选择其他 Mapbox 地图样式
        ),
    showlegend = False
)
    traces.append(trace)
    fig = go.Figure(data=traces, layout=layout)
    fig.write_html('./data/output/track.html')



def main():
    # Load the data from './data/sudden_change_all.xlsx' as ground truth
    data = pd.read_excel('./data/sudden_change_all.xlsx')
    time_start = 2012100500 #请修改为台风生成时间 
    time_end = 2012101912 #请修改为台风消散时间 
    name = 'Prapiroon' #请修改为台风名称
    epoch = int(input('请输入预报轮数: '))

    # Filter the data from time_start to time_end
    data = data[(data['时间'] >= time_start) & (data['时间'] <= time_end) & (data['台风名称'] == name)]

    #load the best track data
    number = int(data['台风编号'].values[0])
    name = data['台风名称'].values[0]
    index = data['数据库文件名'].values[0]

    #apeend the track
    track = None
    angle = None
    sudden_change = []
    for record in data.values:
        # 将时间进行格式转换
        record[3] = datetime.datetime.strptime(str(record[3]), '%Y%m%d%H')
        if angle is None:
            angle = np.array([record[9]])
        else:
            angle = np.append(angle,[record[9]],axis=0)
            if record[9]>45: # sudden change, stored in a index
                sudden_change.append(len(angle)-1)
        if track is None:
            track = np.array([record[3:9]])  #3:时间，4：纬度，5：经度，6：等级，7：气压，8：风速，9：折角
        else:
            track = np.append(track,[record[3:9]],axis=0)

    tc_ground_truth = TC(number,name,index)
    tc_ground_truth.point = len(data)
    tc_ground_truth.track = track
    tc_ground_truth.angle = angle
    tc_ground_truth.sudden_change = sudden_change
    tc_ground_truth.type = 0

    tc_forecast_list = []
    for i in range(1,epoch+1): #进行 epoch 轮 forecast
        file_path = './data/'+str(i)+'/' #第i轮的数据放在文件夹./data/i中
        print(10*'-'+'正在处理第',i,'轮预报'+10*'-')
        time_start_forecast = input('请输入第'+str(i)+'轮预报的起始时间: ')
        forecast_turns = int(input('请输入第'+str(i)+'轮预报的次数: '))
        forecast_interval = int(input('请输入第'+str(i)+'轮预报的时间间隔: '))
        # 查找起报时间对应tc_ground_truth的索引
        index_start = 0
        for j in range(tc_ground_truth.point):
            time = datetime.datetime.strptime(str(tc_ground_truth.track[j][0]), '%Y-%m-%d %H:%M:%S')
            time = time.strftime('%Y%m%d%H')
            if int(time) == int(time_start_forecast):
                index_start = j
                break
        print('起报时间对应tc_ground_truth的索引为:',index_start)
        last_track = np.array([[tc_ground_truth.track[index_start][1],tc_ground_truth.track[index_start][2]]])
        print('起报时间对应经纬度为:','lat:',last_track[0][0],'lon:',last_track[0][1])
        tc_forecast = TC(number,name,index)      
        tc_forecast.type = 1                                                                                                                     
        track = None
        new_track = None
        for j in range(0,forecast_turns+1):
            # 预报数据的文件名格式为：output_{forecast_interval*i}_surface_{time_start_forecast+forecast_interval*i(转换为yymmddhh格式)}.nc
            # 例如：output_6_surface_2012100706.nc
            file_time = datetime.datetime.strptime(time_start_forecast, '%Y%m%d%H') + datetime.timedelta(hours=forecast_interval*j)
            surface_name = 'output_{0}_surface_{1}.nc'.format(forecast_interval*j, file_time.strftime('%Y%m%d%H'))
            upper_name = 'output_{0}_upper_{1}.nc'.format(forecast_interval*j, file_time.strftime('%Y%m%d%H'))
            upper = Dataset(file_path+upper_name)
            surface = Dataset(file_path+surface_name)

            #find the location of the minimum mslp
            lon,lat,pressure,vorticity = find_location_min_mslp(upper,surface,last_track)[0:4]

            #append the track to last_track
            last_track = np.append(last_track,[[lat,lon]],axis=0)
            new_track = np.array([file_time,lat,lon,pressure,vorticity,0])
            if vorticity < 5*10**-5:
                if tc_forecast.false_forecast is None:
                    tc_forecast.false_forecast = np.array([j])
                else:
                    tc_forecast.false_forecast = np.append(tc_forecast.false_forecast,[j],axis=0)
            if track is None:
                track = [new_track]
            else:
                track = np.append(track,[new_track],axis=0)
        tc_forecast.track = track
        tc_forecast_list.append(tc_forecast)
        print(10*'-'+'第',i,'轮预报处理完毕'+10*'-')

    hover_texts = generate_hover_text(tc_ground_truth, tc_forecast_list)
    #plot the track of both ground truth and forecast
    plot_track(tc_ground_truth,tc_forecast_list,hover_texts)
    get_xlsx(tc_ground_truth,tc_forecast_list)

if __name__ == '__main__':
    main()
