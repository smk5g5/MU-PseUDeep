# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:56:54 2019

@author: smk5g5
"""
import pickle
import pandas as pd
import data_processing as dp
#import pirna_kmer as pk
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
import phynet_onehot as pn
from keras.layers import Input
import keras.utils.np_utils as kutils
import keras.models as models
#import threading
import time
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import EarlyStopping
import json
#from sklearn.metrics import matthews_corrcoef
from keras.models import Model

import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
#from sklearn.metrics import accuracy_score, recall_score
remark = ''  # The mark written in the result file.

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)

def read_composite_data(file):
    df = pd.read_csv(file,sep="\s+",names=["Sequence","Chr_loc","Label"])
    return(df)

def subselect_list(indices,mylist):
    final = [mylist[i] for i in indices]
    return  final

def get_all_seq(df):
    seq = []
    label = []
    for indexs in df.index:
        seq.append(df.loc[indexs].values[0])
        label.append(df.loc[indexs].values[2])
    return seq, label


def test_in(cnn,testX, testY, i, argument):
    """
    You can put the val data as a reference adjustment model,
    or you can put the test data evaluation model.
    """
#    cnn = models.load_model('%d-merge.h5' % i, {'isru': isru, 'pearson_r': pearson_r})
    #  ############### test ##########################
    pre_score = cnn.evaluate(testX, testY, batch_size=2048, verbose=0)

    #  ######### Print Precision and Recall ##########
    pred_proba = cnn.predict(testX, batch_size=2048)
    pred_score = pred_proba[:, 1]
    true_class = testY[:, 1]

    precision, recall, _ = precision_recall_curve(true_class, pred_score)
    average_precision = average_precision_score(true_class, pred_score)

    fpr, tpr, thresholds = roc_curve(true_class, pred_score)
    roc_auc = auc(fpr, tpr)

    for index in range(len(pred_score)):
        if pred_score[index] > 0.5:
            pred_score[index] = 1
        else:
            pred_score[index] = 0

    mcc = matthews_corrcoef(true_class, pred_score)

    print('precision:%.3f' % precision_score(y_true=true_class, y_pred=pred_score))
    print('recall:%.3f' % recall_score(y_true=true_class, y_pred=pred_score))
    print('F1:%.3f' % f1_score(y_true=true_class, y_pred=pred_score))

    plt.figure()
    plt.step(recall, precision, color='navy', where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('./plt/' + str(argument) + 'Precision-Recall%d.png' % i)

    #  ################# Print ROC####################

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Inception ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./plt/' + str(argument) + 'ROC %d.png' % i)

    with open('results.txt', 'a') as file:
        file.write('test loss:' + str(pre_score[0]) + '\t' + 'test acc:' + str(pre_score[1]) + '\n' +
                   'precision:%.3f' % precision_score(y_true=true_class, y_pred=pred_score) + '\t'
                   'recall:%.3f' % recall_score(y_true=true_class, y_pred=pred_score) + '\t'
                   'F1:%.3f' % f1_score(y_true=true_class, y_pred=pred_score) + '\n'
                   'mcc:' + str(mcc) + '\t' + 'auc:' + str(roc_auc) + '\n')

    return pre_score


def print_loss(fitHistory, i, argument):
    """print fine tune loss curve"""
    #  ######### Print Loss Map ##########
    plt.figure()
    plt.plot(fitHistory.history['loss'][:-20])  # patience in earlystopping.
    plt.plot(fitHistory.history['val_loss'][:-20])
    # plt.title('size:%d' % size)
    plt.title('LOSS:times %d' % i)
    plt.ylim([0.35, 1.0])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./plt/' + str(argument) + 'loss%d.png' % i)

    #  ############### final ################
    loss1 = fitHistory.history['loss'][-21:-20]
    acc1 = fitHistory.history['binary_accuracy'][-21:-20]
    loss2 = fitHistory.history['val_loss'][-21:-20]
    acc2 = fitHistory.history['val_binary_accuracy'][-21:-20]

    # write results in a file.
    if os.path.exists('results.txt'):
        with open('results.txt', 'a') as file:
            file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\t' + str(i) + remark + '\t' +
                       str(argument) + '\n' + 'train loss:' + str(loss1) + '\t' + 'train acc:' + str(acc1) + '\n' +
                       'val loss:' + str(loss2) + '\t' + 'val acc:' + str(acc2) + '\n')
    else:
        with open('results.txt', 'w') as file:
            file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\t' + str(i) + remark + '\t' +
                       str(argument) + '\n' + 'train loss:' + str(loss1) + '\t' + 'train acc:' + str(acc1) + '\n' +
                       'val loss:' + str(loss2) + '\t' + 'val acc:' + str(acc2) + '\n')

    return loss1, acc1, loss2, acc2    

def run_test_onehot(test,iteration):
    seq_test,label_test = get_all_seq(test)
    testX,testY = dp.onehotkey(seq_test,label_test)
    testY = np_utils.to_categorical(testY,2)
    testY = testY.reshape(-1,2)
    row,col = testX[0].shape
    testX.shape = (testX.shape[0],row,col)
    input = Input(shape=(row,col))
    trained_model = models.load_model('%d-onehot.h5' % iteration)
    pre_score = test_in(trained_model, testX, testY, iteration, 'combined_model')
    print('test loss:', pre_score[0],'test acc:', pre_score[1])

i =0
whole_data = read_composite_data("./combined_human_pseudouridine_seqstr.txt")
whole_data = whole_data.sample(frac=1)
train,test = dp.per_split(whole_data,0.8)
train2,valid = dp.per_split(train,0.8)
seq,label = get_all_seq(train2)
seq_val,label_val = get_all_seq(valid)
trainX, trainY = dp.onehotkey(seq,label)
valX,valY = dp.onehotkey(seq_val,label_val)
class_weight_list = compute_class_weight('balanced', np.unique(trainY), trainY)
class_weights = dict(zip(np.unique(trainY), class_weight_list))
row,col = trainX[0].shape
trainX.shape = (trainX.shape[0], row,col)
input_1 = Input(shape=(row,col))
trainY = np_utils.to_categorical(trainY,2)
cnn, fitHistory = pn.MCNN(trainX, trainY, input_1, i,class_weights)
#print(fitHistory.history)
#exit()
loss1, acc1, loss2, acc2 = print_loss(fitHistory, i, 'one_hot')
print('train loss:', loss1,  'train acc:', acc1,'val loss:', loss2, 'val acc:', acc2)
run_test_onehot(test,i)
