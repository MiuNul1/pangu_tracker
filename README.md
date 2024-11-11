# Pangu Tracking

## 算法说明

* Pangu Tracking 是一项基于针对Pangu-Weather的输出文件进行台风定位的python脚本

* 本项目的算法参考了 [Bi 等，（2023）](https://doi.org/10.1038/s41586-023-06185-3)中提供的定位算法，原文如下

> Algorithm for tracking tropical cyclones
>
> We followed a classical algorithm【38】 that locates the local minimum of MSLP to track the eye of tropical cyclones. Given the starting time point and the corresponding initial position of a cyclone eye, we iteratively called for the 6-hour forecast algorithm and looked for a local minimum of MSLP that satisfies the following conditions:、
>
> •There is a maximum of 850 hPa relative vorticity that is larger than 5 × 10−5 within a radius of 278 km for the Northern Hemisphere, or a minimum that is smaller than −5 × 10−5 for the Southern Hemisphere. 
>
> •There is a maximum of thickness between 850 hPa and 200 hPa within a radius of 278 km when the cyclone is extratropical. 
>
> •The maximum 10-m wind speed is larger than 8 m s−1 within a radius of 278 km when the cyclone is on land. Once the cyclone’s eye is located, the tracking algorithm continued to find the next position in a vicinity of 445 km. The tracking algorithm terminated when no local minimum of MSLP is found to satisfy the above conditions. See Extended Data Fig. 8 for two tracking examples.

* 关于该算法更详细的描述可以参考[Newsletter No. 102 - Winter 2004/05 | ECMWF](https://www.ecmwf.int/en/elibrary/78231-newsletter-no-102-winter-200405)的第8页【Tracking the cyclone】部分
* 本项目属于南京大学大气科学学院本科生大创项目《深度学习模型对台风路径突变预报评估》的项目代码之一，由于只是本科生课题，我们对台风定位的准确性要求不高，因此忽略了上述算法中对相对涡度和厚度的检验，本项目的算法更为简单：
  * 第一个预报时次的定位，需要给定初始场中的TC的经纬度坐标，（我们采用了CMA best track数据）在这一坐标的445km半径内寻找海平面气压最低点（MSLP),作为此时的TC位置
  * 之后所有预报时次，以上一时次的TC位置为中心，寻找445km半径内的海平面气压最低点
* 经过检验，本算法对在海面上的TC定位比较可靠，但对于登陆TC，建议不要使用本算法

## 代码说明

* file_path为需要定位的nc文件存放路径，建议该路径下文件以如下格式命名
filepath|
  output_6_surface_2004092400.nc
  output_6_upper_2004092400.nc
  output_12_surface_2004092406.nc   
  output_12_surface_2004092412.nc
  ...
  output_{forecast_time}_{type}_{time}.nc  
* 本项目通过读取存取CMA best track的xlsx文件来读取TC初始时刻的坐标，请自行更改坐标输入形式
