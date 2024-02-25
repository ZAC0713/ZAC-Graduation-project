'''
import tensorflow.compat.v1 as tf.disable_v2_behavior()
a = tf.constant([5, 2, 4, 3, 1])
b = tf.constant([10, 20, 40, 30, 50])
idx = tf.contrib.framework.argsort(a, direction='DESCENDING')
sorted_a = tf.gather(a, idx)
sorted_b = tf.gather(b, idx)
print(sorted_a)
print(sorted_b)
'''
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import copy
x=np.random.randn(1,5)
y=np.random.randn(5,1)

print(x)
print(y)
z=np.dot(x,y)
print(z)
"""
"""
data=pd.read_csv("D:/Temporal_Relational_Stock_Ranking-master/data/data_HYH.csv")
data_1=data.loc[:,'A']
data_2=data.loc[:,'B']
data_3=data.loc[:,'C']

D=np.empty(825)
E=[1,2.2,4.21,5]


sum=0
sum_1=0
sum_2=0
sum_4=0
count=0
for i in range(1886):
    if data_2[i]>0 and data_2[i]<62876.28:
        D[count]=data_2[i]
        count=count+1
D.sort()
unique_data = np.unique(D)
np.savetxt("D:/Temporal_Relational_Stock_Ranking-master/data/data_HYH - 副本.csv",unique_data)
"""
"""
def combination_sum(nums, target):
    res = []

    def backtrack(nums, target, index, combination):
        if target == 0:
            res.append(list(combination))
            return
        if target < 0 or index >= len(nums):
            return

        for i in range(index, len(nums)):
            if i > index and nums[i] == nums[i - 1]:
                continue

            combination.append(nums[i])
            backtrack(nums, target - nums[i], i + 1, combination)
            combination.pop()

    nums.sort()  # 对数组进行排序，确保结果组合的顺序
    backtrack(nums, target, 0, [])

    return res

result = combination_sum(D,62876.28)
print(result)
"""
"""
for i in range(1886):62876.28
    sum_1=sum_1+data_1[i]
    sum_2 = sum_2 + data_2[i]
    if data_1[i]<0:
        sum_2 = sum_2 - data_1[i]
    else:
        sum_1 = sum_1 + data_1[i]

    if data_2[i]<0:
        sum_1 = sum_1 - data_1[i]
    else:
        sum_2 = sum_2 + data_2[i]


    if data_1[i]!=0:
        sum=sum-data_1[i]
    else:
        sum=sum+data_2[i]

    D[i]="{:.2f}".format(sum)


    if abs(abs(sum)-data_3[i])>0.01:
        print(i)
        sum_4=sum_4+1
        #sum=data_3[i]
        
        print(sum)
        print(truncated_num)
        print(data_1[i])
        print(data_2[i])
        print(data_3[i])
        
        #break
#np.savetxt("D:/Temporal_Relational_Stock_Ranking-master/data/data_HYH - 副本.csv",D)
print(sum_1,sum_2,sum_4)
"""
"""
a=[1,2,3,4,5]
pre=[2,4,3,5,6]
gt=copy.copy(a)
data1=pd.DataFrame(gt)
data1.to_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ gt_btl.csv")
data2=pd.DataFrame(pre)
data2.to_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ gt_btl&pre_btl.csv")

if cur_offset == 240:
    gt = gt_batch
    pre = cur_rr
    data1 = pd.DataFrame(gt)
    data1.to_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ gt_btl.csv")
    data2 = pd.DataFrame(pre)
    data2.to_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/NASDAQ pre_btl.csv")


# 定义两个张量
return_ratio= tf.constant([[11.2],[-30.4], [6.8], [24.6], [13.5]])
x=tf.reduce_max(return_ratio)
y=tf.reduce_min(return_ratio)
z=(return_ratio-y)/(x-y)

#z=tf.sigmoid(tf.scalar_mul(3/x,return_ratio))
all_one = tf.ones([5, 1], dtype=tf.float32)
a = tf.reshape(return_ratio, (5,))
value_1, indices_1 = tf.nn.top_k(a, k=5)
return_ratio_score = value_1[:, tf.newaxis]
pre_pw_dif = tf.subtract(
                tf.matmul(return_ratio_score, all_one, transpose_b=True),
                tf.matmul(all_one, return_ratio_score, transpose_b=True)
            )
rank_loss = tf.reduce_mean(
                tf.nn.relu(

                        tf.multiply(pre_pw_dif, pre_pw_dif),


                )
            )

n=tf.nn.top_k(return_ratio)
all_one=tf.ones([5,],dtype=tf.float32)
x=tf.range(5,dtype=tf.float32)
c1=tf.add(all_one,x)
c=c1
#y=x[:,tf.newaxis]
#c=tf.reshape(y,(5,))
#ground_truth = tf.placeholder(tf.float32, [5, 1])
#values_a,indices_a = tf.nn.top_k(return_ratio, k=5)
#rank_return_ratio= tf.gather(c, indices_a)
values_a,indices_a = tf.nn.top_k(return_ratio, k=5)
y=tf.gather(c, indices_a)
values_b,indices_b = tf.nn.top_k(y, k=5)
y1=tf.gather(c1, indices_b)
v2=tf.reverse(y1,[0])
a=tf.scalar_mul(0.2,v2)

print(x)
with tf.Session() as sess:
    print(sess.run(return_ratio))
    print(sess.run(tf.reduce_max(return_ratio)))

    print(sess.run(x))
    print(sess.run(y))
    print(sess.run(z))
    print(sess.run(return_ratio_score))
    print(sess.run(pre_pw_dif))
    print(sess.run(rank_loss))
sess.close()
print(1/2)

            rank_loss =0.0015*tf.reduce_mean(
                tf.nn.relu(
                    tf.multiply(
                        tf.multiply(pre_pw_dif, gt_pw_dif),
                        mask_pw
                    )
                )
            )


            rank_loss= 0.0015*tf.losses.mean_squared_error(
                ground_truth_score, return_ratio_score, weights=mask
            )
            
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

