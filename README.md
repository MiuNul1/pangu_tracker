# pangu_tracker
对盘古气象大模型输出的nc文件进行TC定位和追踪

## 使用方法：
1. 浏览数据集，确定你想要进行起报的时间
2. 用盘古模型以起报时间对应的era5为输入进行预报（e.g.预报步长3h,预报次数24次）
3. 将模型的输入nc文件分别命名为0_surface.nc,0_upper.nc
4. 将模型的输出nc文件按照预报时间依次命名为3_surface.nc，6_surface.nc...... 72_surface.nc以及3_upper.nc,6_upper.nc...... 72_upper.nc
5. 分别将上述nc文件存放在surface目录和upper目录下
6. 打开tracker.py，在第141行和142行分别设置好time_start已经time_end (time_start 即最终绘制的tc路径图中besttrack 初始的时间， time_end即最终绘制的tc路径图中besttrack 结束的时间，不宜超过你所预报的tc在best track最后一条记录的时间)
