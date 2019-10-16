"""
# -*- coding: utf-8 -*-
# @Time    : 2019/5/17 11:42
# @Author  : smk5g5
# @Email   : smk5g5@mail.missouri.edu
# @File    : transfer learning pseudo-uridine
"""


import pickle
import pandas as pd
import data_processing as dp
#import pirna_kmer as pk
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
#import phy_net as pn
from keras.layers import Input
import keras.utils.np_utils as kutils
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
import time
import numpy as np
import matplotlib
import pickle
matplotlib.use('Agg')
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler, History
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1_l2
import keras.metrics
import matplotlib.pyplot as plt
from keras.optimizers import Nadam, Adam
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
import data_processing
import os
from sklearn import svm
from sklearn.manifold import TSNE
from matplotlib import offsetbox
from sklearn.metrics import accuracy_score, recall_score
from keras.optimizers import Nadam,Adam,RMSprop,SGD
import bootstrap_hyperas_model_weights as bt
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)

def run_test_onehot(test,iteration,argument):
    seq_test,secstr_test,label_test = get_all_seq(test)
    testX2,testY = onehotkey(seq_test,label_test)
    testX1,testY = onehotkey_sec(secstr_test,label_test)
    testY = np_utils.to_categorical(testY,2)
    testY = testY.reshape(-1,2)
    row1,col1 = testX1[0].shape
    row2,col2 = testX2[0].shape
    testX1.shape = (testX1.shape[0],row1,col1)
    testX2.shape = (testX2.shape[0],row2,col2)
##    testx1_out = "./testx1_"+str(i)
##    testx2_out = "./testx2_"+str(i)
##    testy_out = "./testy_"+str(i)
##    pickle.dump(testX1, open(testx1_out, "wb" ))
##    pickle.dump(testX2, open(testx2_out, "wb" ))
##    pickle.dump(testY, open(testy_out, "wb" ))
    pre_score = test_in(testX1,testX2, testY, iteration, argument)
    print('test loss:', pre_score[0],'test acc:', pre_score[1])

def per_split(df, rate):
    """

    :param df:
    :param rate:
    :return:
    """
    df_new = df.sample(frac=1)
    df_train = df_new[0: int(len(df_new) * rate)]
    df_test = df_new[int(len(df_new) * rate): -1]
    return df_train, df_test

def read_composite_data(file):
    df = pd.read_csv(file,sep="\s+",names=["Sequence","Secondary_str","Chr_loc","Label"])
    return(df)

def subselect_list(indices,mylist):
    final = [mylist[i] for i in indices]
    return  final

def get_all_seq(df):
    seq = []
    secstr = []
    label = []
    for indexs in df.index:
        seq.append(df.loc[indexs].values[0])
        secstr.append(df.loc[indexs].values[1])
        label.append(df.loc[indexs].values[3])
    return seq,secstr,label

def onehotkey(seq, tag):
     """
     one hot coding
     :param seq:
     :param tag:
     :return:
     """
     tag = np.array(tag)
#     for num in range(len(seq)):
#         seq[num] = seq[num].strip('\n')
     letterDict = {}
     letterDict["A"] = 0
     letterDict["C"] = 1
     letterDict["G"] = 2
     letterDict["U"] = 3
     letterDict["T"] = 3
     CategoryLen = 4
     probMatr = np.zeros((len(seq),len(seq[0]), CategoryLen))
     sampleNo = 0
     for sequence in seq:
         RNANo = 0
         for RNA in sequence:
             try:
                 index = letterDict[RNA]
                 probMatr[sampleNo][RNANo][index] = 1
                 RNANo += 1
             except:
                 RNANo += 1
         sampleNo += 1
     return probMatr, tag

#          'H' => 4186861, (Nucleotides in Hairpin context)
#          'F' => 2584467, (Nucleotides in dangling start shape)
#          'M' => 1081741, (Nucleotides in multiloops)
#          'T' => 2595203, (Nucleotides in dangling ends)
#          'S' => 10667944, (Nucleotides in Stems)
#          'I' => 2968228 (Nucleotides in interntal loops)
def onehotkey_sec(seq, tag):
     """
     one hot coding
     :param seq:
     :param tag:
     :return:
     """
     tag = np.array(tag)
#     for num in range(len(seq)):
#         seq[num] = seq[num].strip('\n')
     letterDict = {}
     letterDict["H"] = 0
     letterDict["F"] = 1
     letterDict["M"] = 2
     letterDict["S"] = 3
     letterDict["T"] = 4
     letterDict["I"] = 5
     CategoryLen = 6
     probMatr = np.zeros((len(seq),len(seq[0]), CategoryLen))
     sampleNo = 0
     for sequence in seq:
         RNANo = 0
         for RNA in sequence:
             try:
                 index = letterDict[RNA]
                 probMatr[sampleNo][RNANo][index] = 1
                 RNANo += 1
             except:
                 RNANo += 1
         sampleNo += 1
     return probMatr, tag


def fine_tuning(trainX1,trainX2,trainY1,valX1,valX2,valY1,input_1,input_2,i,class_weights,t):
    if t==0:
        layer_names = ['0_secstr','1_secstr','0','1','dense_concat']
        cnn = models.load_model("/home/smk5g5/emboss_60_perc_data/8-secstr_seq_denseconcat_60perc_best.h5")
        for layer in cnn.layers:
            if layer.name in layer_names:
                layer.trainable=True
            else:
                layer.trainable=False
##        nadam = Nadam(lr=0.00001)
        sgd = SGD(lr=0.008615890670714792)
        cnn.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[keras.metrics.binary_accuracy])
        checkpointer = ModelCheckpoint(filepath='%d-yeast_pretrain-merge.h5' % i, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=2000)
        fitHistory = cnn.fit([trainX1,trainX2], trainY1, batch_size=32, nb_epoch=500,validation_data=([valX1,valX2], valY1),callbacks=[checkpointer,early_stopping],class_weight=class_weights)
        myjson_file = "myhist_" +"dict_" + "secstr_seq_denseconcat_60perc_trial26_" +str(i)
        json.dump(fitHistory.history, open(myjson_file, 'w'))
        return cnn,fitHistory
    else:
        cnn = models.load_model('%d-yeast_pretrain-merge.h5' % i)
        checkpointer = ModelCheckpoint(filepath='%d-yeast_pretrain-merge.h5' % i, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=2000)
        fitHistory = cnn.fit([trainX1,trainX2], trainY1, batch_size=32, nb_epoch=500,validation_data=([valX1,valX2], valY1),callbacks=[checkpointer,early_stopping],class_weight=class_weights)
        myjson_file = "myhist_" +"dict_" + "secstr_seq_denseconcat_60perc_trial26_" +str(i)
        json.dump(fitHistory.history, open(myjson_file, 'a'))
        return cnn,fitHistory


def test_in(testX1,testX2, testY, i,argument):
    """
    You can put the val data as a reference adjustment model,
    or you can put the test data evaluation model.
    """
    cnn = models.load_model('%d-yeast_pretrain-merge.h5' % i)
    #  ############### test ##########################
    pre_score = cnn.evaluate([testX1,testX2], testY, batch_size=32, verbose=0)

    #  ######### Print Precision and Recall ##########
    pred_proba = cnn.predict([testX1,testX2], batch_size=32)
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
        file.write('#############'+str(argument)+'#################'+'\n')
        file.write('test loss:' + str(pre_score[0]) + '\t' + 'test acc:' + str(pre_score[1]) + '\n' +
                   'precision:%.3f' % precision_score(y_true=true_class, y_pred=pred_score) + '\t'
                   'recall:%.3f' % recall_score(y_true=true_class, y_pred=pred_score) + '\t'
                   'F1:%.3f' % f1_score(y_true=true_class, y_pred=pred_score) + '\n'
                   'mcc:' + str(mcc) + '\t' + 'auc:' + str(roc_auc) + '\n')
        file.write('##############################'+'\n')

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

##yeast_emboss_60perc_test_0.txt
##yeast_emboss_60perc_train_6.txt
def main():
    for i in range(10):
        train_file = 'mouse_emboss_60perc_train_'+str(i)+'.txt'
        test_file =  'mouse_emboss_60perc_testing_'+str(i)+'.txt'
        valid_file = 'mouse_emboss_60perc_validation_'+str(i)+'.txt'
        train = read_composite_data(train_file)
        train = train.sample(frac=1)
        valid = read_composite_data(valid_file)
        valid = valid.sample(frac=1)
        test = read_composite_data(test_file)
        test = test.sample(frac=1)
        seq_val,secstr_val,label_val = get_all_seq(valid)
        ValX2, ValY = onehotkey(seq_val, label_val)
        ValX1,ValY = onehotkey_sec(secstr_val, label_val)
        ValY = np_utils.to_categorical(ValY,2)
        ValY = ValY.reshape(-1,2)
        pos_train = train.loc[train['Label']==1]
        pos_len = int((pos_train.shape[0])*0.9)
        neg_train = train.loc[train['Label']==0]
        neg_pos_ratio = neg_train.shape[0]/len(pos_train)
        class_weights = {0: 1,
                    1:1}
        pos_test = test.loc[test['Label'] == 1]
        neg_test = test.loc[test['Label'] == 0]
        rand_neg = neg_test.sample(pos_test.shape[0])
        test_new = pos_test.append(rand_neg)
        test_new = test_new.sample(frac=1)
        for t in range(int(neg_pos_ratio)):
            train_neg = neg_train[(pos_len*t):((pos_len*t)+pos_len)]
            train_all = pd.concat([pos_train.sample(pos_len),train_neg])
            train_all = train_all.sample(frac=1)
            seq,secstr,label = get_all_seq(train_all)
            trainX2, trainY = onehotkey(seq, label)
            trainX1, trainY = onehotkey_sec(secstr, label)
            row1,col1 = trainX1[0].shape
            input_1 = Input(shape=(row1,col1))
            trainX1.shape = (trainX1.shape[0], row1,col1)
            row2,col2 = trainX2[0].shape
            trainX2.shape = (trainX2.shape[0], row2,col2)
            input_2 = Input(shape=(row2,col2))
            trainY = np_utils.to_categorical(trainY,2)
            trainY = trainY.reshape(-1,2)
            print(trainX1.shape,"\t",trainX2.shape,"\t",trainY.shape,"\t",ValX1.shape,"\t",ValX2.shape,"\t",ValY.shape)
            cnn, fitHistory = fine_tuning(trainX1,trainX2,trainY,ValX1,ValX2,ValY,input_1,input_2,i,class_weights,t)
        run_test_onehot(test,i,'imbalanced_test_hyperas_best')
        run_test_onehot(test_new,i,'balanced_test_hyperas_best')
        outfile_test = 'balanced_' + 'mouse_emboss_60perc_testing_' + str(i) +".txt"
        test_new.to_csv(outfile_test,sep='\t',index=False,header=False)

if __name__ == '__main__':
    main()
