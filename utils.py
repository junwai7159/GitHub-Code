import numpy as np
import csv
import tensorflow as tf
import platform
import os
from scipy import interpolate
import random
import re

class DataControl:
    def __init__(self, filepath):
        self.files = os.listdir(filepath)
        x, y, self.user_ID, seq_lens = self.loadfile(filepath)
        padded_x, padded_y, self.seq_lens = self.padding_data(x, seq_lens, y)
        self.train_idx, self.test_idx = self.indexsplit(len(x), israndom=False)

        self.x_train = padded_x[self.train_idx]
        self.y_train = padded_y[self.train_idx]
        self.seq_len_train = self.seq_lens[self.train_idx]

        self.x_test = padded_x[self.test_idx]
        self.y_test = padded_y[self.test_idx]
        self.seq_len_test = self.seq_lens[self.test_idx]
        self.user_test = self.user_ID[self.test_idx] 

    def indexsplit(self,indexlength,israndom):
        if israndom is True:
            randomind = list(range(indexlength))
            np.random.shuffle(randomind)
            trainindex = randomind[:int(len(randomind) * 0.9)]
            testindex = list(filter(lambda j: j not in trainindex, list(randomind)))
        else:
            trainindex=[]
            testindex =[]
            for i in range(indexlength):
                if self.user_ID[i] in [16, 17, 18]:
                    testindex.append(i)
                else:
                    if self.user_ID[i] <= 32:
                        trainindex.append(i)
        return trainindex,testindex

    def loadfile(self,filepath):
        motion = []
        heart = []
        label = []
        datalen = []
        rawindex = []
        userID = []
        for file in self.files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if len(res) >= 1 and int(res[0]) < 43 or int(res[0]) >= 3000 or int(res[0]) in [50, 51, 52, 53]:
                filename = filepath + file
                data = np.load(filename, allow_pickle=True)
                data = data['datapre']
                select_ind =[0, 1, 7, 8]
                motion_data = data[:, select_ind]
                motion_data2 = []
                #计算第5个特征，是求一个窗口内呼吸频率的方差
                for i in range(len(motion_data)):
                    if i < 2:
                        tmp = motion_data[i].tolist()
                        tmp.extend([
                               np.log10(np.var(motion_data[i:i+3, 3])+0.000001)/10])
                        motion_data2.append(tmp)
                    else:
                        if i > len(motion_data)-3:
                            tmp = motion_data[i].tolist()
                            tmp.extend([
                                np.log10(np.var(motion_data[i-2:i+1, 3])+0.000001)/10])
                            motion_data2.append(tmp)
                        else:
                            tmp = motion_data[i].tolist()
                            tmp.extend([
                                        np.log10(np.var(motion_data[i-1:i+2, 3])+0.000001)/10])
                            motion_data2.append(tmp)
                motion_data2 = np.array(motion_data2)
                #将第5个特征进行归一化处理
                fftvar = motion_data2[:, 4]
                mfftvar = np.mean(fftvar)
                for i in range(len(motion_data2)):
                    motion_data2[i][4] = motion_data2[i][4]/mfftvar
                label_data = data[:, -1]
                label_data = label_data.tolist()
                for i in range(len(label_data)):
                    if label_data[i] == 2:
                        label_data[i] = 1
                    if label_data[i] == 4:
                        label_data[i] = 3
                motion.append(motion_data2)
                label.append(np.array(label_data))
                userID.append(int(res[0]))
                datalen.append(len(motion_data2))
        return motion, label, np.array(userID), datalen

    def padding_data(self, data, slen, ys):
        lengths = slen
        max_length = max(lengths)
        num_samples = len(lengths)
        padding_data = np.zeros([num_samples, max_length, 5])
        padding_data[:, :, :] = padding_data[:, :, :]
        for idx, seq in enumerate(data):
            padding_data[idx, :len(seq), :] = seq
        padding_label = np.zeros([num_samples, max_length]) + 6
        for idx, seq in enumerate(ys):
            padding_label[idx, :len(seq)] = seq

        return padding_data, padding_label, np.array(slen)
