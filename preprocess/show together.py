import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
#规定格式
#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False

#读取时，由于前几个训练轮次的误差较大，故将数据舍去不再图中展示
data_1=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/ret.csv",skiprows=[1,2,3,4])
data_2=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/ret rank loss.csv",skiprows=[1,2,3,4])

#设置横坐标值，也就是训练轮次
xdata=np.arange(1,47)
#读取文件中的指定列
#ret.csv
ydata_1_1=data_1.loc[:,'# Train Loss']
ydata_1_2=data_1.loc[:,'Valid MSE']
ydata_1_3=data_1.loc[:,'Valid_pr_btl']
ydata_1_4=data_1.loc[:,'Test MSE']
ydata_1_5=data_1.loc[:,'Test_pr_btl']
ydata_1_6=data_1.loc[:,'Better valid loss']
#ret rank loss.csv
ydata_2_1=data_2.loc[:,'# Train Loss']
ydata_2_2=data_2.loc[:,'Valid MSE']
ydata_2_3=data_2.loc[:,'Valid_pr_btl']
ydata_2_4=data_2.loc[:,'Test MSE']
ydata_2_5=data_2.loc[:,'Test_pr_btl']
ydata_2_6=data_2.loc[:,'Better valid loss']


#——————————————————————————————————————————————————————————
#以下三个在一个图中展示
fig=plt.figure(figsize=(33, 8), dpi=45)
fig_dims = (1, 3)
#Train loss
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata, ydata_1_1, c='red',linewidth=2,label="relation rank loss")
plt.plot(xdata, ydata_2_1, c='green',linewidth=2,label="rank loss")
plt.scatter(xdata, ydata_1_1, c='red',s=15)
plt.scatter(xdata, ydata_2_1, c='green',s=15)
plt.legend(loc='best',prop = {'size':18})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=21)  # 设置x轴刻度字体大小
plt.yticks(fontsize=21)  # 设置y轴刻度字体大小
plt.title(u"Train loss Comparison",size=23)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=23)   #设置x轴名为“x轴名”
plt.ylabel(u'Value of Train loss',size=22)   #设置y轴名为“y轴名”

#Valid MSE
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_1_2, c='red',linewidth=2,label="relation rank loss")
plt.plot(xdata, ydata_2_2, c='green',linewidth=2,label="rank loss")
plt.scatter(xdata, ydata_1_2, c='red',s=15)
plt.scatter(xdata, ydata_2_2, c='green',s=15)
plt.legend(loc='best',prop = {'size':18})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=21)  # 设置x轴刻度字体大小
plt.yticks(fontsize=21)  # 设置y轴刻度字体大小
plt.title(u"Valid MSE Comparison",size=23)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=23)   #设置x轴名为“x轴名”
plt.ylabel(u'Value of Valid MSE',size=22)   #设置y轴名为“y轴名”

#Test MSE
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 2))
plt.plot(xdata, ydata_1_3, c='red',linewidth=2,label="relation rank loss")
plt.plot(xdata, ydata_2_3, c='green',linewidth=2,label="rank loss")
plt.scatter(xdata, ydata_1_3, c='red',s=15)
plt.scatter(xdata, ydata_2_3, c='green',s=15)
plt.legend(loc='best',prop = {'size':18})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=21)  # 设置x轴刻度字体大小
plt.yticks(fontsize=21)  # 设置y轴刻度字体大小
plt.title(u"Test MSE Comparison",size=23)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=23)   #设置x轴名为“x轴名”
plt.ylabel(u'Value of Test MSE',size=22)   #设置y轴名为“y轴名”

plt.show()


#——————————————————————————————————————————————
#以下两个在一个图中展示
fig=plt.figure(figsize=(22, 8), dpi=55)
fig_dims = (1, 2)
#Valid btl
plt.subplots_adjust(wspace=0.2)#两图上下间距
plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata, ydata_1_3, c='red',linewidth=1.3,label="relation rank loss")
plt.plot(xdata, ydata_2_3, c='green',linewidth=1.3,label="rank loss")
plt.scatter(xdata, ydata_1_3, c='red',s=13)
plt.scatter(xdata, ydata_2_3, c='green',s=13)
plt.legend(loc='best',prop = {'size':16})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=18)  # 设置x轴刻度字体大小
plt.yticks(fontsize=18)  # 设置y轴刻度字体大小
plt.title(u"Valid performance btl Comparison",size=21)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=21)   #设置x轴名为“x轴名”
#Test btl
plt.subplots_adjust(wspace=0.2)#两图上下间距
plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_1_5, c='red',linewidth=1.3,label="relation rank loss")
plt.plot(xdata, ydata_2_5, c='green',linewidth=1.3,label="rank loss")
plt.scatter(xdata, ydata_1_5, c='red',s=13)
plt.scatter(xdata, ydata_2_5, c='green',s=13)
plt.legend(loc='best',prop = {'size':16})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=18)  # 设置x轴刻度字体大小
plt.yticks(fontsize=18)  # 设置y轴刻度字体大小
plt.title(u"Test performance btl Comparison",size=21)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=21)   #设置x轴名为“x轴名”
plt.show()


#Best btl
plt.figure(figsize=(15, 8), dpi=50)
plt.plot(xdata, ydata_1_6, c='red',linewidth=1.5,label="relation rank loss")
plt.plot(xdata, ydata_2_6, c='green',linewidth=1.5,label="rank loss")
plt.scatter(xdata, ydata_1_6, c='red',s=13)
plt.scatter(xdata, ydata_2_6, c='green',s=13)
plt.legend(loc='best',prop = {'size':17})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=20)  # 设置x轴刻度字体大小
plt.yticks(fontsize=20)  # 设置y轴刻度字体大小
plt.title(u"Best Valid loss Comparison",size=23)   #设置表名为“表名”
plt.xlabel(u'Training rounds',size=21)   #设置x轴名为“x轴名”
plt.show()
