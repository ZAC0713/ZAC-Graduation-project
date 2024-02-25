import argparse
import copy
import numpy as np
import os
import random
import tensorflow as tf
import datetime
import pandas as pd
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops


    def leaky_relu(features, alpha=0.2, name=None):
        with ops.name_scope(name, "LeakyRelu", [features, alpha]):
            features = ops.convert_to_tensor(features, name="features")
            alpha = ops.convert_to_tensor(alpha, name="alpha")
            return math_ops.maximum(alpha * features, features)


    def pprint(*args):
        time = '[' + str(datetime.datetime.utcnow() + datetime.timedelta(hours=8))[:19] + ']-'
        print(time, *args, flush=True)
        global_log_file = "/time_preformance.txt"
        with open(global_log_file, 'a') as f:
            print(time, *args, flush=True, file=f)

from load_data import load_EOD_data
from evaluator import evaluate

class RankLSTM:
    def __init__(self, data_path, market_name, tickers_fname, parameters,
                 steps=1, epochs=50, batch_size=None, gpu=True):
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        # load data
        self.tickers = np.genfromtxt(os.path.join("D:/Temporal_Relational_Stock_Ranking-master/data/NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv", '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        ### DEBUG
        # self.tickers = self.tickers[0: 10]
        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data("D:/Temporal_Relational_Stock_Ranking-master/data/2013-01-01", market_name, self.tickers, steps)

        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )


    def train(self):
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()

            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 1])
            mask = tf.placeholder(tf.float32, [self.batch_size, 1])
            feature = tf.placeholder(tf.float32,
                [self.batch_size, self.parameters['seq'], self.fea_dim])
            base_price = tf.placeholder(tf.float32, [self.batch_size, 1])
            all_one = tf.ones([self.batch_size, 1], dtype=tf.float32)
            '''
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.parameters['unit']
            )
            '''
            lstm_cell = tf.contrib.rnn.GRUCell(
                self.parameters['unit']
            )
            initial_state = lstm_cell.zero_state(self.batch_size,
                                                 dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(
                lstm_cell, feature, dtype=tf.float32,
                initial_state=initial_state
            )

            seq_emb = outputs[:, -1, :]
            # One hidden layer
            prediction = tf.layers.dense(
                seq_emb, units=1, activation=leaky_relu, name='reg_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            return_ratio = tf.div(tf.subtract(prediction, base_price), base_price)
            reg_loss = tf.losses.mean_squared_error(
                ground_truth, return_ratio, weights=mask
            )
            """
            x = tf.reduce_max(return_ratio)
            return_ratio_score = tf.sigmoid(tf.scalar_mul(2.75 / x, return_ratio))
            y = tf.reduce_max(ground_truth)
            ground_truth_score = tf.sigmoid(tf.scalar_mul(2.75 / y, ground_truth))
            
            #对预测的收益率和真实收益率进行降维，（1026，1）——》(1026,)
            a=tf.reshape(return_ratio,(self.batch_size,))
            b=tf.reshape(ground_truth,(self.batch_size,))
            value_1,indices_1=tf.nn.top_k(a,k=self.batch_size)
            value_2, indices_2 = tf.nn.top_k(b, k=self.batch_size)
            ground_truth_score=value_2[:,tf.newaxis]
            return_ratio_score=value_1[:,tf.newaxis]
           
            #设置一个全1张量，与c1这个从0循环到1025的张量相加，变为从1到1026的c2,再复制一个c3=c2，一个在真实值里使用，一个在预测值里使用
            c1=tf.range(self.batch_size,dtype=tf.float32)
            func_one=tf.ones([self.batch_size,],dtype=tf.float32)
            c2=tf.add(c1,func_one)
            c3=c2
            c4=c2
            c5=c2
            c6=c2
            #让收益率张量与c2同步降序排序
            values_a, indices_a = tf.nn.top_k(a, k=self.batch_size)
            x1=tf.gather(c2, indices_a)
            values_a1, indices_a1 = tf.nn.top_k(x1, k=self.batch_size)
            x3= tf.gather(c3, indices_a1)
            #将上一步c2与a同步排序得出的张量x1翻转，然后等比例缩小self.batch_size倍，最后增加维度，输出
            y1=tf.reverse(x3,[0])
            z1=tf.scalar_mul(1/self.batch_size, y1)
            return_ratio_score=z1[:,tf.newaxis]

            values_b, indices_b = tf.nn.top_k(b, k=self.batch_size)
            x2=tf.gather(c3, indices_b)
            values_b1, indices_b1 = tf.nn.top_k(x2, k=self.batch_size)
            x4= tf.gather(c4, indices_b1)
            y2= tf.reverse(x4, [0])
            z2= tf.scalar_mul(1/self.batch_size, y2)
            ground_truth_score=z2[:,tf.newaxis]
            
            pre_pw_dif = tf.subtract(
                tf.matmul(return_ratio_score, all_one, transpose_b=True),
                tf.matmul(all_one, return_ratio_score, transpose_b=True)
            )
            gt_pw_dif = tf.subtract(
                tf.matmul(all_one, ground_truth_score, transpose_b=True),
                tf.matmul(ground_truth_score, all_one, transpose_b=True)
            )
            mask_pw = tf.matmul(mask, mask, transpose_b=True)

            rank_loss = tf.reduce_mean(
                tf.nn.relu(
                    tf.multiply(
                        tf.multiply(pre_pw_dif, gt_pw_dif),
                        mask_pw
                    )
                )
            )
            """
            loss =reg_loss
            #loss = reg_loss
            #tf.cast(self.parameters['alpha'], tf.float32) *

            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.parameters['lr']
            ).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        best_valid_pred = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_gt = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_mask = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_test_pred = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_gt = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_mask = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_valid_perf = {
            'mse': np.inf, 'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrrt': 0.0,
            'btl': 0.0, 'abtl': 0.0, 'btl5': 0.0, 'abtl5': 0.0, 'btl10': 0.0,
            'abtl10': 0.0, 'rho': -1.0
        }
        best_test_perf = {
            'mse': np.inf, 'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrrt': 0.0,
            'btl': 0.0, 'abtl': 0.0, 'btl5': 0.0, 'abtl5': 0.0, 'btl10': 0.0,
            'abtl10': 0.0, 'rho': -1.0
        }
        best_valid_loss = np.inf

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        list=np.zeros((50,16))
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            #tra_rank_loss = 0.0

            for j in range(self.valid_index - self.parameters['seq'] -
                           self.steps + 1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j])
                feed_dict = {
                    feature: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss,cur_reg_loss,batch_out = \
                    sess.run((loss, reg_loss,optimizer),
                             feed_dict)
                tra_loss += cur_loss
                tra_reg_loss += cur_reg_loss
                #tra_rank_loss += cur_rank_loss

            list[i][0]=tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1)
            list[i][1]=tra_reg_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1)
            list[i][2]=0
                #tra_rank_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1)
            print('Train Loss:',
                  tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  #tra_rank_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1)
                  )


            cur_valid_pred = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_gt = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_mask = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            val_loss = 0.0
            val_reg_loss = 0.0
            #val_rank_loss= 0.0
            for cur_offset in range(
                self.valid_index - self.parameters['seq'] - self.steps + 1,
                self.test_index - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss,cur_semb, cur_rr, = \
                    sess.run((loss, reg_loss,seq_emb,
                              return_ratio), feed_dict)

                val_loss += cur_loss
                val_reg_loss += cur_reg_loss
                #val_rank_loss += cur_rank_loss
                cur_valid_pred[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(cur_rr[:, 0])
                cur_valid_gt[:, cur_offset - (self.valid_index -
                                              self.parameters['seq'] -
                                              self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_valid_mask[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
                """
                if cur_offset == 756+218:
                    gt = np.sort(gt_batch[:, 0])[::-1]
                    pre = np.sort(cur_rr[:, 0])[::-1]
                    data1 = pd.DataFrame(gt)
                    data1.to_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/1_NASDAQ gt&&pre btl and btl rank/1_NASDAQ valid/NASDAQ gt_btl.csv")
                    data2 = pd.DataFrame(pre)
                    data2.to_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/1_NASDAQ gt&&pre btl and btl rank/1_NASDAQ valid/NASDAQ pre_btl.csv")
                """
            list[i][3]=val_loss / (self.test_index - self.valid_index)
            list[i][4] = val_reg_loss / (self.test_index - self.valid_index)
            list[i][5] = 0
                #val_rank_loss / (self.test_index - self.valid_index)
            print('Valid MSE:',
                  val_loss / (self.test_index - self.valid_index),
                  val_reg_loss / (self.test_index - self.valid_index),
                  #val_rank_loss / (self.test_index - self.valid_index)
                  )

            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt,
                                      cur_valid_mask)
            list[i][6]=cur_valid_perf['mse']
            list[i][7]=cur_valid_perf['mrrt']
            list[i][8]=cur_valid_perf['btl']
            print('\t Valid preformance:', cur_valid_perf)

            # test on testing set
            cur_test_pred = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_gt = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_mask = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            test_loss = 0.0
            test_reg_loss = 0.0
            #test_rank_loss = 0.0
            for cur_offset in range(
                self.test_index - self.parameters['seq'] - self.steps + 1,
                self.trade_dates - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss,cur_reg_loss,cur_semb, cur_rr = \
                    sess.run((loss, reg_loss, seq_emb,
                              return_ratio), feed_dict)

                test_loss += cur_loss
                test_reg_loss += cur_reg_loss
                #test_rank_loss += cur_rank_loss
                cur_test_pred[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(cur_rr[:, 0])
                cur_test_gt[:, cur_offset - (self.test_index -
                                             self.parameters['seq'] -
                                             self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_test_mask[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
                """
                if cur_offset == 756+230+204:
                    gt = np.sort(gt_batch[:, 0])[::-1]
                    pre = np.sort(cur_rr[:, 0])[::-1]
                    data1 = pd.DataFrame(gt)
                    data1.to_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/1_NASDAQ gt&&pre btl and btl rank/2_NASDAQ test/NASDAQ gt_btl.csv")
                    data2 = pd.DataFrame(pre)
                    data2.to_csv("D:/Temporal_Relational_Stock_Ranking-master/data/result/1_NASDAQ gt&&pre btl and btl rank/2_NASDAQ test/NASDAQ pre_btl.csv")
                """
            list[i][9]=test_loss / (self.trade_dates - self.test_index)
            list[i][10] = test_reg_loss / (self.trade_dates - self.test_index)
            list[i][11] = 0
                #test_rank_loss / (self.trade_dates - self.test_index)
            print('Test MSE:',
                  test_loss / (self.trade_dates - self.test_index),
                  test_reg_loss / (self.trade_dates - self.test_index),
                  #test_rank_loss / (self.trade_dates - self.test_index)
                  )

            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            list[i][12]=cur_test_perf['mse']
            list[i][13]=cur_test_perf['mrrt']
            list[i][14]=cur_test_perf['btl']
            print('\t Test performance:', cur_test_perf)
            # if cur_valid_perf['mse'] < best_valid_perf['mse']:
            if val_loss / (self.test_index - self.valid_index) < \
                    best_valid_loss:
                best_valid_loss = val_loss / (self.test_index -
                                              self.valid_index)
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_gt = copy.copy(cur_valid_gt)
                best_valid_pred = copy.copy(cur_valid_pred)
                best_valid_mask = copy.copy(cur_valid_mask)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_gt = copy.copy(cur_test_gt)
                best_test_pred = copy.copy(cur_test_pred)
                best_test_mask = copy.copy(cur_test_mask)

                print('Better valid loss:', best_valid_loss)
                list[i][15]=best_valid_loss
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))

        #np.savetxt("D:/Temporal_Relational_Stock_Ranking-master/data/result/NYSE btl1/1_NYSE rank loss(only mean square loss).csv", list,delimiter=',', header='Train Loss,Train reg loss,Train rank loss,Valid Loss,Valid reg loss,Valid rank loss,Valid_pr_mse,Valid_pr_mrrt,Valid_pr_btl,Test Loss,Test reg loss,Test rank loss,Test_pr_mse,Test_pr_mrrt,Test_pr_btl,Better valid loss')
        print('\nBest Valid performance:', best_valid_perf)
        print('\tBest Test performance:', best_test_perf)
        sess.close()
        tf.reset_default_graph()

        return best_valid_pred, best_valid_gt, best_valid_mask, \
               best_test_pred, best_test_gt, best_test_mask

    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True


if __name__ == '__main__':
    desc = 'train a rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.001,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='use gpu')
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('parameters:', parameters)

    rank_LSTM = RankLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        parameters=parameters,
        steps=1, epochs=1, batch_size=None, gpu=args.gpu
    )
    pred_all = rank_LSTM.train()
