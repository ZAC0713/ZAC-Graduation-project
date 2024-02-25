import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
#规定格式
#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False

#读取时，由于前几个训练轮次的误差较大，故将数据舍去不再图中展示
#NASDAQ市场使用排序损失函数与不同排序损失函数对比
data_1=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ btl10/3_NASDAQ relation rank loss(modified max value ranking loss).csv",skiprows=[1,2,3,4])
data_2=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ btl10/4_NASDAQ relation rank loss(modified max abs ranking loss).csv",skiprows=[1,2,3,4])
data_3=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ btl10/5_NASDAQ relation rank loss(modified normalized ranking loss).csv",skiprows=[1,2,3,4])
data_4=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ btl10/2_NASDAQ relation rank loss(original ranking loss).csv",skiprows=[1,2,3,4])
data_5=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ btl10/1_NASDAQ relation rank loss(only mean square loss).csv",skiprows=[1,2,3,4])
data_10=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ btl10/3_NASDAQ relation rank loss(modified max value ranking loss).csv",skiprows=[1])
#NASDAQ和NYSE市场LSTM与relation LSTM的对比
data_6=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ btl1/1_NASDAQ rank loss(only mean square loss).csv",skiprows=[1,2,3,4])
data_7=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ btl1/6_NASDAQ relation rank loss(sort pair distance loss).csv",skiprows=[1,2,3,4])
data_8=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NYSE btl1/1_NYSE rank loss(only mean square loss).csv",skiprows=[1,2,3,4])
data_9=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NYSE btl1/6_NYSE relation rank loss(sort pair distance loss).csv",skiprows=[1,2,3,4])

#NYSE市场的使用排序损失函数对比
data_11=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NYSE btl10/1_NYSE relation rank loss(only mean square loss-50 epoch).csv",skiprows=[1,2,3,4])
data_12=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NYSE btl10/2_NYSE relation rank loss(max value ranking loss-50 epoch).csv",skiprows=[1,2,3,4])
data_13=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NYSE btl10/3_NYSE relation rank loss(max abs ranking loss-50 epoch).csv",skiprows=[1,2,3,4])
#设置横坐标值，也就是训练轮次
xdata=np.arange(1,47)
xdata_1=np.arange(1,50)
xdata_2=np.arange(1,47)
#为对比损失函数和归一化方式做准备
ydata_1=data_1.loc[:,'Valid_pr_btl']
ydata_2=data_2.loc[:,'Valid_pr_btl']
ydata_3=data_3.loc[:,'Valid_pr_btl']
ydata_4=data_1.loc[:,'Test_pr_btl']
ydata_5=data_2.loc[:,'Test_pr_btl']
ydata_6=data_3.loc[:,'Test_pr_btl']
ydata_7=data_4.loc[:,'Valid_pr_btl']
ydata_8=data_4.loc[:,'Test_pr_btl']
ydata_9=data_5.loc[:,'Valid_pr_btl']
ydata_10=data_5.loc[:,'Test_pr_btl']

#为对比NASDAQ市场LSTM与relationLSTM做准备
ydata_11=data_6.loc[:,'Valid_pr_btl']
ydata_12=data_6.loc[:,'Test_pr_btl']
ydata_13=data_7.loc[:,'Valid_pr_btl']
ydata_14=data_7.loc[:,'Test_pr_btl']

#为对比NASDAQ市场LSTM与relationLSTM做准备
ydata_15=data_8.loc[:,'Valid_pr_btl']
ydata_16=data_8.loc[:,'Test_pr_btl']
ydata_17=data_9.loc[:,'Valid_pr_btl']
ydata_18=data_9.loc[:,'Test_pr_btl']

#NASDAQ三种归一化方法的MRR对比
ydata_19=data_1.loc[:,'Valid_pr_mrrt']
ydata_20=data_2.loc[:,'Valid_pr_mrrt']
ydata_21=data_3.loc[:,'Valid_pr_mrrt']
ydata_22=data_1.loc[:,'Test_pr_mrrt']
ydata_23=data_2.loc[:,'Test_pr_mrrt']
ydata_24=data_3.loc[:,'Test_pr_mrrt']
#NASDAQ三种损失函数的MRR对比
ydata_28=data_1.loc[:,'Test_pr_mrrt']
ydata_29=data_4.loc[:,'Test_pr_mrrt']
ydata_30=data_5.loc[:,'Test_pr_mrrt']

#两个损失函数的loss
ydata_25=data_10.loc[:,'Train rank loss']
ydata_26=data_10.loc[:,'Valid rank loss']
ydata_27=data_10.loc[:,'Test rank loss']

ydata_31=data_10.loc[:,'Train reg loss']
ydata_32=data_10.loc[:,'Valid reg loss']
ydata_33=data_10.loc[:,'Test reg loss']

#NYSE是否使用最大值压缩归一化排序损失函数对比
ydata_34=data_11.loc[:,'Valid_pr_btl']
ydata_35=data_12.loc[:,'Valid_pr_btl']
ydata_42=data_13.loc[:,'Valid_pr_btl']

ydata_36=data_11.loc[:,'Test_pr_btl']
ydata_37=data_12.loc[:,'Test_pr_btl']
ydata_43=data_13.loc[:,'Test_pr_btl']

ydata_38=data_11.loc[:,'Valid_pr_mrrt']
ydata_39=data_12.loc[:,'Valid_pr_mrrt']
ydata_44=data_13.loc[:,'Valid_pr_mrrt']

ydata_40=data_11.loc[:,'Test_pr_mrrt']
ydata_41=data_12.loc[:,'Test_pr_mrrt']
ydata_45=data_13.loc[:,'Test_pr_mrrt']

"""
#三种归一化收益率对比
fig=plt.figure(figsize=(18, 6),dpi=100)
fig_dims = (1, 2)
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata, ydata_1, '-',c='#00C957',linewidth=1.5,label="max value normalization",)
plt.plot(xdata, ydata_2, '-',c='r',linewidth=1.5,label="max abs normalization")
plt.plot(xdata, ydata_3, '-',c='#1E90FF',linewidth=1.8,label="standard normalization")
plt.scatter(xdata, ydata_1, c='#00C957',s=5)
plt.scatter(xdata, ydata_2, c='r',s=5)
plt.scatter(xdata, ydata_3, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':15.5})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of three normalization in Valid",size=17)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl10',size=15)   #设置y轴名为“y轴名”

plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_4, '-',c='#00C957',linewidth=1.7,label="max value normalization",)
plt.plot(xdata, ydata_5, '-',c='r',linewidth=1.5,label="max abs normalization")
plt.plot(xdata, ydata_6, '-',c='#1E90FF',linewidth=1.8,label="standard normalization")
plt.scatter(xdata, ydata_4, c='#00C957',s=5)
plt.scatter(xdata, ydata_5, c='r',s=5)
plt.scatter(xdata, ydata_6, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':15.5})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of three normalization in Test",size=17)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl10',size=15)   #设置y轴名为“y轴名”
plt.show()



#三种排序损失函数收益率对比
fig=plt.figure(figsize=(18, 6),dpi=100)
fig_dims = (1, 2)
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 0))

plt.plot(xdata, ydata_1, '-',c='#00C957',linewidth=1.5,label="max value normalization",)
plt.plot(xdata, ydata_7, '-',c='r',linewidth=1.5,label="use rate as ranking loss")
plt.plot(xdata, ydata_9, '-',c='#1E90FF',linewidth=1.8,label="only mean square loss")
plt.scatter(xdata, ydata_1, c='#00C957',s=5)
plt.scatter(xdata, ydata_7, c='r',s=5)
plt.scatter(xdata, ydata_9, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':15.5})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of three ranking loss in Valid",size=17)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl10',size=15)   #设置y轴名为“y轴名”

plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_4, '-',c='#00C957',linewidth=1.5,label="max value normalization",)
plt.plot(xdata, ydata_8, '-',c='r',linewidth=1.5,label="use rate as ranking loss")
plt.plot(xdata, ydata_10, '-',c='#1E90FF',linewidth=1.8,label="only mean square loss")
plt.scatter(xdata, ydata_4, c='#00C957',s=5)
plt.scatter(xdata, ydata_8, c='r',s=5)
plt.scatter(xdata, ydata_10, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':15.5})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of three ranking loss in Test",size=17)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl10',size=15)   #设置y轴名为“y轴名”
plt.show()


#LSTM与relation LSTM 在NASDAQ的收益率对比
fig=plt.figure(figsize=(18, 6),dpi=100)
fig_dims = (1, 2)
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 0))

plt.plot(xdata, ydata_11, '-',c='#00C957',linewidth=1.7,label="LSTM",)
plt.plot(xdata, ydata_13, '-',c='#1E90FF',linewidth=1.8,label="Dynamic GCN-LSTM")
plt.scatter(xdata, ydata_11, c='#00C957',s=5)
plt.scatter(xdata, ydata_13, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':14})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of two model in Valid",size=18)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl1',size=15)   #设置y轴名为“y轴名”

plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_12, '-',c='#00C957',linewidth=1.7,label="LSTM",)
plt.plot(xdata, ydata_14, '-',c='#1E90FF',linewidth=1.8,label="Dynamic GCN-LSTM")
plt.scatter(xdata, ydata_12, c='#00C957',s=5)
plt.scatter(xdata, ydata_14, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':14})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of two model in Test",size=18)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl1',size=15)   #设置y轴名为“y轴名”
plt.show()

#LSTM与relation LSTM 在NYSE的收益率对比
fig=plt.figure(figsize=(18, 6),dpi=100)
fig_dims = (1, 2)
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 0))

plt.plot(xdata, ydata_15, '-',c='#00C957',linewidth=1.7,label="LSTM",)
plt.plot(xdata, ydata_17, '-',c='#1E90FF',linewidth=1.8,label="Dynamic GCN-LSTM")
plt.scatter(xdata, ydata_15, c='#00C957',s=5)
plt.scatter(xdata, ydata_17, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':15})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of two model in Valid",size=18)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl1',size=15)   #设置y轴名为“y轴名”

plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata, ydata_16, '-',c='#00C957',linewidth=1.7,label="LSTM",)
plt.plot(xdata, ydata_18, '-',c='#1E90FF',linewidth=1.8,label="Dynamic GCN-LSTM")
plt.scatter(xdata, ydata_16, c='#00C957',s=5)
plt.scatter(xdata, ydata_18, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':15})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of two model in Test",size=18)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl1',size=15)   #设置y轴名为“y轴名”
plt.show()

#最大值压缩归一化的reg_loss对比
fig=plt.figure(figsize=(21, 4),dpi=100)
fig_dims = (1, 3)
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata_1, ydata_31, '-',c='#00C957',linewidth=1.5)
plt.scatter(xdata_1, ydata_31, c='#00C957',s=5)
#plt.legend(loc='best',prop = {'size':13})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Train reg loss in NASDAQ",size=15)   #设置表名为“表名”
#plt.xlabel(u'training rounds',size=19)   #设置x轴名为“x轴名”

plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata_1, ydata_32, '-',c='#00C957',linewidth=1.5)
plt.scatter(xdata_1, ydata_32, c='#00C957',s=5)
#plt.legend(loc='best',prop = {'size':13})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Valid reg loss in NASDAQ",size=15)   #设置表名为“表名”
#plt.xlabel(u'training rounds',size=19)   #设置x轴名为“x轴名”

plt.subplot2grid(fig_dims, (0, 2))
plt.plot(xdata_1, ydata_33, '-',c='#00C957',linewidth=1.5)
plt.scatter(xdata_1, ydata_33, c='#00C957',s=5)
#plt.legend(loc='best',prop = {'size':13})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Test reg loss in NASDAQ",size=15)   #设置表名为“表名”
#plt.xlabel(u'training rounds',size=19)   #设置x轴名为“x轴名”
plt.show()

#最大值压缩归一化的rank_loss对比
fig=plt.figure(figsize=(21, 4),dpi=100)
fig_dims = (1, 3)
plt.subplots_adjust(wspace=0.3)#两图上下间距
plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata_1, ydata_25, '-',c='#1E90FF',linewidth=1.5)
plt.scatter(xdata_1, ydata_25, c='#1E90FF',s=5)
#plt.legend(loc='best',prop = {'size':13})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Train ranking loss in NASDAQ",size=15)   #设置表名为“表名”
#plt.xlabel(u'training rounds',size=19)   #设置x轴名为“x轴名”

plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata_1, ydata_26, '-',c='#1E90FF',linewidth=1.5)
plt.scatter(xdata_1, ydata_26, c='#1E90FF',s=5)
#plt.legend(loc='best',prop = {'size':13})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Valid ranking loss in NASDAQ",size=15)   #设置表名为“表名”
#plt.xlabel(u'training rounds',size=19)   #设置x轴名为“x轴名”

plt.subplot2grid(fig_dims, (0, 2))
plt.plot(xdata_1, ydata_27, '-',c='#1E90FF',linewidth=1.5)
plt.scatter(xdata_1, ydata_27, c='#1E90FF',s=5)
#plt.legend(loc='best',prop = {'size':13})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Test ranking loss in NASDAQ",size=15)   #设置表名为“表名”
#plt.xlabel(u'training rounds',size=19)   #设置x轴名为“x轴名”
plt.show()


#NASDAQ上三种损失函数的MRR比较
plt.figure(figsize=(21, 6),dpi=100)
plt.plot(xdata, ydata_28, '-',c='#00C957',linewidth=1.9,label="modified ranking loss",)
plt.plot(xdata, ydata_29, '-',c='r',linewidth=1.9,label="ordinary ranking loss")
plt.plot(xdata, ydata_30, '-',c='#1E90FF',linewidth=1.9,label="only mean square loss")
plt.scatter(xdata, ydata_28, c='#00C957',s=5)
plt.scatter(xdata, ydata_29, c='r',s=5)
plt.scatter(xdata, ydata_30, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':18})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=15)  # 设置y轴刻度字体大小
plt.title(u"Comparison of three loss's MRR in Test",size=23)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=22)   #设置x轴名为“x轴名”
plt.ylabel(u'MRR',size=20)   #设置y轴名为“y轴名"
plt.show()
"""
#NYSE是否使用最大绝对值压缩归一化排序感知损失函数的收益率对比
fig=plt.figure(figsize=(18, 6),dpi=100)
fig_dims = (1, 2)
plt.subplots_adjust(wspace=0.3)#两图上下间距

plt.subplot2grid(fig_dims, (0, 1))
plt.plot(xdata_2, ydata_43, '-',c='#00C957',linewidth=1.5,label="max abs ranking loss")
plt.plot(xdata_2, ydata_37, '-',c='r',linewidth=1.5,label="max value ranking loss")
plt.plot(xdata_2, ydata_36, '-',c='#1E90FF',linewidth=1.7,label="only mean square loss",)
plt.scatter(xdata_2, ydata_43, c='#00C957',s=5)
plt.scatter(xdata_2, ydata_37, c='r',s=5)
plt.scatter(xdata_2, ydata_36, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':17})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of two loss in Test",size=20)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl10',size=15)   #设置y轴名为“y轴名”

plt.subplot2grid(fig_dims, (0, 0))
plt.plot(xdata_2, ydata_42, '-',c='#00C957',linewidth=1.85,label="max abs ranking loss")
plt.plot(xdata_2, ydata_35, '-',c='r',linewidth=1.5,label="max value ranking loss")
plt.plot(xdata_2, ydata_34, '-',c='#1E90FF',linewidth=1.7,label="only mean square loss")
plt.scatter(xdata_2, ydata_42, c='#00C957',s=5)
plt.scatter(xdata_2, ydata_35, c='r',s=5)
plt.scatter(xdata_2, ydata_34, c='#1E90FF',s=5)
plt.legend(loc='best',prop = {'size':17})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=13)  # 设置x轴刻度字体大小
plt.yticks(fontsize=13)  # 设置y轴刻度字体大小
plt.title(u"Comparison of two loss in Valid",size=20)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=17)   #设置x轴名为“x轴名”
plt.ylabel(u'btl10',size=15)   #设置y轴名为“y轴名”
plt.show()

#NYSE是否使用最大值压缩归一化排序感知损失函数的MRR对比
plt.figure(figsize=(21, 6),dpi=100)
plt.plot(xdata_2, ydata_44, '-',c='#00C957',linewidth=1.7,label="max abs ranking loss")
plt.plot(xdata_2, ydata_39, '-',c='r',linewidth=1.7,label="max value ranking loss")
plt.plot(xdata_2, ydata_38, '-',c='#1E90FF',linewidth=1.7,label="only mean square loss")
plt.scatter(xdata_2, ydata_44, c='#00C957',s=6)
plt.scatter(xdata_2, ydata_39, c='r',s=6)
plt.scatter(xdata_2, ydata_38, c='#1E90FF',s=6)
plt.legend(loc='best',prop = {'size':20})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=15)  # 设置y轴刻度字体大小
plt.title(u"Comparison of two loss's MRR in Valid",size=23)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=22)   #设置x轴名为“x轴名”
plt.ylabel(u'MRR',size=20)   #设置y轴名为“y轴名”
plt.show()

plt.figure(figsize=(21, 6),dpi=100)
plt.plot(xdata_2, ydata_45, '-',c='#00C957',linewidth=1.7,label="max abs ranking loss")
plt.plot(xdata_2, ydata_41, '-',c='r',linewidth=1.7,label="max value ranking loss")
plt.plot(xdata_2, ydata_40, '-',c='#1E90FF',linewidth=1.7,label="only mean square loss")
plt.scatter(xdata_2, ydata_45, c='#00C957',s=6)
plt.scatter(xdata_2, ydata_41, c='r',s=6)
plt.scatter(xdata_2, ydata_40, c='#1E90FF',s=6)
plt.legend(loc='best',prop = {'size':20})
plt.grid(True, linestyle='--', alpha=0.5,)
plt.xticks(fontsize=15)  # 设置x轴刻度字体大小
plt.yticks(fontsize=15)  # 设置y轴刻度字体大小
plt.title(u"Comparison of two loss's MRR in Test",size=23)   #设置表名为“表名”
plt.xlabel(u'training rounds',size=22)   #设置x轴名为“x轴名”
plt.ylabel(u'MRR',size=20)   #设置y轴名为“y轴名”
plt.show()


