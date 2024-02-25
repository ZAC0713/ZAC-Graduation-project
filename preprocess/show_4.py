import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from pylab import xticks,yticks,np
#规定格式
#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False

data_1=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/1_NASDAQ gt&&pre btl and btl rank/2_NASDAQ test/副本1/NASDAQ gt_btl.csv")
data_2=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/1_NASDAQ gt&&pre btl and btl rank/2_NASDAQ test/副本2/NASDAQ gt_btl.csv")
data_3=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/1_NASDAQ gt&&pre btl and btl rank/2_NASDAQ test/副本3/NASDAQ gt_btl.csv")


xdata=np.arange(1,1027)

ydata_1=data_1.loc[:,'0']
ydata_2=data_2.loc[:,'0']
ydata_3=data_3.loc[:,'0']

fig=plt.figure(figsize=(20, 5.25),dpi=88)
fig_dims = (1, 3)
plt.subplots_adjust(wspace=0.25)#三图左右间距

plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata, ydata_1, '-',c='#00C957',linewidth=1.6,label="day 1",)
plt.scatter(xdata, ydata_1, c='#00C957',s=1.2)
plt.legend(loc='best',prop = {'size':20})
plt.grid(True, linestyle='--', alpha=0.5,)
#xticks_labels =['0','350','700','1050','1400','1737']
#xticks(np.linspace(0,1737,6,endpoint=True),xticks_labels,fontsize=13)
plt.xticks(fontsize=16)  # 设置x轴刻度字体大小
plt.yticks(fontsize=16)  # 设置y轴刻度字体大小
plt.xlabel(u'stock number',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'ground truth',size=17)   #设置y轴名为“y轴名"

plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_2, '-',c='#1E90FF',linewidth=1.6,label="day 2")
plt.scatter(xdata, ydata_2, c='#1E90FF',s=1.2)
plt.legend(loc='best',prop = {'size':20})
plt.grid(True, linestyle='--', alpha=0.5,)
#xticks_labels =['0','350','700','1050','1400','1737']
#xticks(np.linspace(0,1737,6,endpoint=True),xticks_labels,fontsize=13)
plt.xticks(fontsize=16)  # 设置x轴刻度字体大小
plt.yticks(fontsize=16)  # 设置y轴刻度字体大小
plt.xlabel(u'stock number',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'ground truth',size=17)   #设置y轴名为“y轴名"

plt.subplot2grid(fig_dims, (0, 2))
plt.plot(xdata, ydata_3, '-',c='r',linewidth=1.6,label="day 3")
plt.scatter(xdata, ydata_3, c='r',s=1.2)
plt.legend(loc='best',prop = {'size':20})
plt.grid(True, linestyle='--', alpha=0.5,)
#xticks_labels =['0','350','700','1050','1400','1737']
#xticks(np.linspace(0,1737,6,endpoint=True),xticks_labels,fontsize=13)
plt.xticks(fontsize=16)  # 设置x轴刻度字体大小
plt.yticks(fontsize=16)  # 设置y轴刻度字体大小
plt.xlabel(u'stock number',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'ground truth',size=17)   #设置y轴名为“y轴名"




plt.show()