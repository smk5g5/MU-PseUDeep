# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:35:18 2019

@author: smk5g5
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:15:16 2019

@author: smk5g5

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
import bootstrap_hyperas_model_weights as bt
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
from sklearn.metrics import confusion_matrix

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)


#    physical_code_x = core.Flatten()(input_1)
#    physical_code_x = BatchNormalization()(physical_code_x)
#    physical_code_x = Dense(1024, kernel_initializer='glorot_normal', activation='softplus', name='7')(physical_code_x)
#    physical_code_x = BatchNormalization()(physical_code_x)
#    physical_code_x = Dropout(0.2)(physical_code_x)
#    physical_code_x = Dense(512, kernel_initializer='glorot_normal', activation='softplus', name='8')(physical_code_x)
#    physical_code_x = BatchNormalization()(physical_code_x)
#    physical_code_x = Dropout(0.4)(physical_code_x)
#    physical_code_x = Dense(256, kernel_initializer='glorot_normal', activation='softplus', name='9')(physical_code_x)
#    physical_code_x = BatchNormalization()(physical_code_x)
#    physical_code_x = Dropout(0.5)(physical_code_x)
#    output_physical_x = Dense(128, kernel_initializer='glorot_normal', activation='relu', name='10')(physical_code_x)

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


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds,argument):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.savefig('./plt/' + str(argument) + 'decision_threshold%d.png' % i)

def test_in2(testX1,testX2, testY, i,thresh,argument):
    cnn = models.load_model('%d-secstr_seq_denseconcat_60perc_best.h5' % i)
    pre_score = cnn.evaluate([testX1,testX2], testY, batch_size=32, verbose=0)
    pred_proba = cnn.predict([testX1,testX2], batch_size=32)
    pred_score = pred_proba[:, 1]
    true_class = testY[:, 1]
    precision, recall, thresholds = precision_recall_curve(true_class, pred_score)
    plot_precision_recall_vs_threshold(precision,recall,thresholds,argument)
    average_precision = average_precision_score(true_class, pred_score)
    fpr, tpr, thresholds = roc_curve(true_class, pred_score)
    roc_auc = auc(fpr, tpr)
    for index in range(len(pred_score)):
        if pred_score[index] > thresh:
            pred_score[index] = 1
        else:
            pred_score[index] = 0
    mcc = matthews_corrcoef(true_class, pred_score)
    tn, fp, fn, tp = confusion_matrix(true_class, pred_score).ravel()
    specificity =  tn / (tn+fp)
    sensitivity = tp/(tp+fn)
    print('Threshold:',thresh,'sensitivity:',sensitivity,'specificity:',specificity)
#def test_in(testX1,testX2, testY, i,thresh,argument):
#    """
#    You can put the val data as a reference adjustment model,
#    or you can put the test data evaluation model.
#    """
#    cnn = models.load_model('%d-secstr_seq_denseconcat_60perc_best.h5' % i)
#    #  ############### test ##########################
#    pre_score = cnn.evaluate([testX1,testX2], testY, batch_size=32, verbose=0)
#
#    #  ######### Print Precision and Recall ##########
#    pred_proba = cnn.predict([testX1,testX2], batch_size=32)
#    pred_score = pred_proba[:, 1]
#    true_class = testY[:, 1]
#
#    precision, recall, thresholds = precision_recall_curve(true_class, pred_score)
#    average_precision = average_precision_score(true_class, pred_score)
#
#    fpr, tpr, thresholds = roc_curve(true_class, pred_score)
#    roc_auc = auc(fpr, tpr)
#
#    for index in range(len(pred_score)):
#        if pred_score[index] > thresh:
#            pred_score[index] = 1
#        else:
#            pred_score[index] = 0
#
#    mcc = matthews_corrcoef(true_class, pred_score)
#
#    print('precision:%.3f' % precision_score(y_true=true_class, y_pred=pred_score))
#    print('recall:%.3f' % recall_score(y_true=true_class, y_pred=pred_score))
#    print('F1:%.3f' % f1_score(y_true=true_class, y_pred=pred_score))
#
#    plt.figure()
#    plt.step(recall, precision, color='navy', where='post')
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.grid(True)
#    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
#    plt.savefig('./plt/' + str(argument) + 'Precision-Recall%d.png' % i)
#
#    #  ################# Print ROC####################
#
#    plt.figure()
#    lw = 2
#    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Inception ROC curve (area = %0.2f)' % roc_auc)
#    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic')
#    plt.legend(loc="lower right")
#    plt.savefig('./plt/' + str(argument) + 'ROC %d.png' % i)
#
#    with open('results.txt', 'a') as file:
#        file.write('#############'+str(argument)+'#################'+'\n')
#        file.write('test loss:' + str(pre_score[0]) + '\t' + 'test acc:' + str(pre_score[1]) + '\n' +
#                   'precision:%.3f' % precision_score(y_true=true_class, y_pred=pred_score) + '\t'
#                   'recall:%.3f' % recall_score(y_true=true_class, y_pred=pred_score) + '\t'
#                   'F1:%.3f' % f1_score(y_true=true_class, y_pred=pred_score) + '\n'
#                   'mcc:' + str(mcc) + '\t' + 'auc:' + str(roc_auc) + '\n')
#        file.write('##############################'+'\n')
#    return pre_score


def run_test_onehot(test,iteration,argument,thresh):
    seq_test,secstr_test,label_test = get_all_seq(test)
    testX2,testY = onehotkey(seq_test,label_test)
    testX1,testY = onehotkey_sec(secstr_test,label_test)
    testY = np_utils.to_categorical(testY,2)
    testY = testY.reshape(-1,2)
    row1,col1 = testX1[0].shape
    row2,col2 = testX2[0].shape
    testX1.shape = (testX1.shape[0],row1,col1)
    testX2.shape = (testX2.shape[0],row2,col2)
    test_in2(testX1,testX2, testY, iteration,thresh,argument)
#    print('test loss:', pre_score[0],'test acc:', pre_score[1])

thresh_scores = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
i = 0
imbalanced_test_file = "emboss_60perc_testing_"+str(i)+".txt"
balanced_test_file = "balanced_emboss_60perc_testing_"+str(i)+".txt"
test_imbal = read_composite_data(imbalanced_test_file)
test_bal = read_composite_data(balanced_test_file)
for thresh in thresh_scores:
    arg_imbal = 'imbalanced_test_'+str(thresh)
    arg_bal = 'balanced_test_'+str(thresh)
    run_test_onehot(test_imbal,i,arg_imbal,thresh)
    run_test_onehot(test_bal,i,arg_bal,thresh)