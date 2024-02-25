import matplotlib.pyplot as plt
from pylab import xticks,yticks,np
import csv
import pandas as pd
import sklearn
import sklearn.preprocessing
import numpy as np
#规定格式
plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False

df1= pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/1/MSFT_1.csv",index_col = 0)
df2= pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/1/GOOGL.csv",index_col = 0)
print(df1.head())
#min-max归一化，将开盘价，收盘价，最高价，最低价归一化
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()#调用sklearn的最小最大值归一化函数MinMaxScaler()
    df['High']= min_max_scaler.fit_transform(df.High.values.reshape(-1,1))#reshape(-1,1)表示将数据转换成1列,然后再进行归一化
    return df#将归一化的结果返回给df



df_m=normalize_data(df1)
df_g=normalize_data(df2)
plt.figure(figsize=(12,5),dpi=90)
xticks_labels =['January','March','May','July','September','December']
xticks(np.linspace(0,252,6,endpoint=True),xticks_labels)
plt.plot(df_m.High.values,c="red",label="微软",linewidth=1.5)
plt.plot(df_g.High.values,c="green",label="谷歌",linewidth=1.5,linestyle='--')
plt.legend(loc='best',prop = {'size':12})
plt.title(u"行业关联示例",size=15)   #设置表名为“表名”
plt.xlabel(u'[2016/01/04--2016/12/30]',size=12)   #设置x轴名为“x轴名”
plt.ylabel(u'股票收盘价格',size=15)   #设置y轴名为“y轴名”
plt.show()