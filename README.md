# pangu_tracker
对盘古气象大模型输出的nc文件进行TC定位和追踪

## 使用方法：
### 输入设置
1. 浏览数据集，确定你想要进行起报的时间
2. 用盘古模型以起报时间对应的era5为输入进行预报（e.g.预报步长3h,预报次数24次）
3. 将第一轮预报的所有output文件目录下的文件全部剪贴到data/1目录下
4. 将用于第一轮预报的input文件改名为output_0_surface_（你的起报时间）和output_0_upper_（你的起报时间）后剪贴到data/1目录下，如下图所示(特别注意要删除.nc前的空格)

   ![image](https://github.com/MiuNul1/pangu_tracker/assets/119723303/1fc55bb5-fdf0-493a-a9c7-7e7fd9b81985)
   
5.第二轮预报同理。有多少轮预报就建立多少个文件夹
6.至此，你的输入文件已经全部整理好了，下面来设置脚本。

### 脚本设置
1.将脚本的第153行至155行全部改成你所预报的TC的信息,代码块中的格式供参考
```
    time_start = 2012100500 #请修改为台风生成时间 
    time_end = 2012101912 #请修改为台风消散时间 
    name = 'Prapiroon' #请修改为台风名称
```
2.运行脚本，根据提示输入数据

![image](https://github.com/MiuNul1/pangu_tracker/assets/119723303/1f997e57-25d5-47bf-91ab-0f292a89971d)

第n轮预报的起始时间指的是第n个文件夹中output_0_upper(surface)_后面的字符串
预报次数即盘古进行了多少次预报
时间间隔指相邻两次预报中的时间间隔（小时）
