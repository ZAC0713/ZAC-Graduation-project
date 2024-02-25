import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
#规定格式
#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False

#读取时，由于前几个训练轮次的误差较大，故将数据舍去不再图中展示
data=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NYSE btl10/4_NYSE relation rank loss(max abs ranking loss-100 epoch).csv",skiprows=[1])

#设置横坐标值，也就是训练轮次
xdata=np.arange(1,100)
#读取文件中的每一列

ydata=data.loc[:,'Train rank loss']
ydata_1=data.loc[:,'Valid Loss']
ydata_2=data.loc[:,'Valid reg loss']
ydata_3=data.loc[:,'Valid rank loss']
ydata_4=data.loc[:,'Valid_pr_btl']
ydata_5=data.loc[:,'Test Loss']
ydata_6=data.loc[:,'Test reg loss']
ydata_7=data.loc[:,'Test rank loss']
ydata_8=data.loc[:,'Test_pr_btl']
ydata_9=data.loc[:,'Better valid loss']
"""
#——————————————————————————————————————————————————————————
#以下三个在一个图中展示

#train loss
fig=plt.figure(figsize=(33, 8), dpi=45)
fig_dims = (1, 3)
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata, ydata, c='#1E90FF',linewidth=2)
plt.scatter(xdata, ydata, c='#1E90FF',s=10)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=21)  # 设置x轴刻度字体大小
plt.yticks(fontsize=21)  # 设置y轴刻度字体大小
plt.title(u"Train rank loss in NYSE",size=23)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=23)   #设置x轴名为“x轴名”
#plt.ylabel(u'Value of Train loss',size=22)   #设置y轴名为“y轴名”

#vaild loss
plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_3, c='#1E90FF',linewidth=2)
plt.scatter(xdata, ydata_3, c='#1E90FF',s=10)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=20.5)  # 设置x轴刻度字体大小
plt.yticks(fontsize=20.5)  # 设置y轴刻度字体大小
plt.title(u"Valid rank loss in NYSE",size=23)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=23)   #设置x轴名为“x轴名”
#plt.ylabel(u'Value of Valid MSE',size=22)   #设置y轴名为“y轴名”

#test Loss
plt.subplot2grid(fig_dims, (0, 2))
plt.plot(xdata, ydata_7, c='#1E90FF',linewidth=2)
plt.scatter(xdata, ydata_7, c='#1E90FF',s=10)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=21)  # 设置x轴刻度字体大小
plt.yticks(fontsize=21)  # 设置y轴刻度字体大小
plt.title(u"Test rank loss in NYSE",size=23)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=23)   #设置x轴名为“x轴名”
#plt.ylabel(u'Value of Test MSE',size=22)   #设置y轴名为“y轴名”
plt.show()
#_______________________________________________________________
"""
#以下两个在一个图中展示
fig=plt.figure(figsize=(20, 5.3), dpi=70)
fig_dims = (1, 2)

#Valid btl10
plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata, ydata_4, c='r',linewidth=1.3)
plt.scatter(xdata, ydata_4, c='r',s=2)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Valid btl10 in NYSE",size=15)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=15)   #设置x轴名为“x轴名”

#Test btl10
plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_8, c='r',linewidth=1.3)
plt.scatter(xdata, ydata_8, c='r',s=2)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Test btl10 in NYSE",size=15)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=15)   #设置x轴名为“x轴名”

plt.show()

#——————————————————————————————————————————————————————
#以下两个在一个图中展示
fig=plt.figure(figsize=(20, 7), dpi=60)
fig_dims = (1, 2)

#test reg loss
plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata, ydata_6, c='red',linewidth=1)
plt.scatter(xdata, ydata_6, c='red',s=13)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Test reg loss",size=15)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=15)   #设置x轴名为“x轴名”

#test rank loss
plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_7, c='blue',linewidth=1)
plt.scatter(xdata, ydata_7, c='blue',s=13)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=18)  # 设置x轴刻度字体大小
plt.yticks(fontsize=18)  # 设置y轴刻度字体大小
plt.title(u"Test rank loss",size=21)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=21)   #设置x轴名为“x轴名

plt.show()

#————————————————————————————————————————————
#以下两个在一个图中展示
fig=plt.figure(figsize=(20, 7), dpi=60)
fig_dims = (1, 2)

#valid_pr_btl
plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata, ydata_4, c='red',linewidth=1)
plt.scatter(xdata, ydata_4, c='red',s=13)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=18)  # 设置x轴刻度字体大小
plt.yticks(fontsize=18)  # 设置y轴刻度字体大小
plt.title(u"Valid performance btl",size=21)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=21)   #设置x轴名为“x轴名”

#test_pr_btl
plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_8, c='blue',linewidth=1)
plt.scatter(xdata, ydata_8, c='blue',s=13)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=18)  # 设置x轴刻度字体大小
plt.yticks(fontsize=18)  # 设置y轴刻度字体大小
plt.title(u"Test performance btl",size=21)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=21)   #设置x轴名为“x轴名”

plt.show()

#Best_valid_loss
plt.figure(figsize=(15, 7), dpi=60)
plt.plot(xdata, ydata_9, c='green',linewidth=1.5)
plt.scatter(xdata, ydata_9, c='green',s=15)
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=18)  # 设置x轴刻度字体大小
plt.yticks(fontsize=18)  # 设置y轴刻度字体大小
plt.title(u"Best Valid Loss",size=22)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=19)   #设置x轴名为“x轴名”
plt.show()

