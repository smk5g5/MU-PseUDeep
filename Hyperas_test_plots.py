# -*- coding: utf-8 -*-
"""
Spyder Editor

Visualize hyperas imbalanced test results for 100 trials

"""
import pickle
import pandas as pd
import data_processing as dp
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Input
import keras.utils.np_utils as kutils
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
from keras.models import Model

import tensorflow as tf
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
remark = ''  # The mark written in the result file.
import time
import numpy as np
import pickle
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
from keras.optimizers import Nadam, Adam
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
import data_processing
import os
from sklearn import svm
from sklearn.manifold import TSNE
from matplotlib import offsetbox
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 7.0)
import seaborn as sns

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)


# pred_proba = cnn.predict([testX1,testX2]
y_real = []
y_proba = []

# colors = cycle(['cyan', 'crimson', 'seagreen', 'yellow', 'blue', 'darkorange', 'darkviolet', 'fuchsia',
#                'deepskyblue', 'lightcoral'])

sns.set(font_scale = 1.2)
colors = sns.hls_palette(101)
fig, ax = plt.subplots()
fig.set_size_inches(12.0, 8.0)

lw = 2
i = 0
precision_rnn = []
recall_rnn = []
threshold_rnn = []
f1_all = []
mcc_all = []

file1 = open("./imbalanced_testset/testx1_0", 'rb')
testX1 = pickle.load(file1)
file1 = open("./imbalanced_testset/testx2_0", 'rb')
testX2 = pickle.load(file1)
file1 = open("./imbalanced_testset/testy_0", 'rb')
testY = pickle.load(file1)

#balanced_test/testx1_0  balanced_test/testx2_0  balanced_test/testy_0
#imbalanced_testset/testx1_0  imbalanced_testset/testx2_0  imbalanced_testset/testy_0
#62-
def get_acceptable_f1scores(input_file):
    import re
    fin = open(input_file,encoding="utf8")
    counter = 0
    mydict_hyperas_trialf1 = dict()
    lines = (line.rstrip() for line in fin) # All lines including the blank ones
    for line in lines:
        string_s = line.strip()
        if "%" in string_s:
            continue
        elif '--' in string_s:
            continue
        elif 'F1 score:' in string_s:
            counter = counter + 1
        elif(re.findall("\d+\.\d+",string_s)):
            mydict_hyperas_trialf1[counter] = float(string_s)
        else:
             continue
    acceptable_fscores = []
    count = 0
    for i in mydict_hyperas_trialf1:
        if(mydict_hyperas_trialf1[i]==0):
            continue
        else:
            acceptable_fscores.append(i)
            count+=1
    top_ten_f1 = top_ten_hyperas_trials(mydict_hyperas_trialf1,acceptable_fscores)
    return(top_ten_f1)

def top_ten_hyperas_trials(mydict_hyperas_trialf1,acceptable_fscores):
    F1_scores = []
    for i in acceptable_fscores:
        F1_scores.append(mydict_hyperas_trialf1[i])
    F1_scores.sort(reverse=True)
    top_ten_hyperas_trials= []
    for i in acceptable_fscores:
        for f1 in F1_scores[:10]:
            if(f1==mydict_hyperas_trialf1[i]):
                print(i,mydict_hyperas_trialf1[i])
                top_ten_hyperas_trials.append(i)
                continue
            else:
                continue
    return(list(set(top_ten_hyperas_trials)))
    
hyperas_trialf1_list = get_acceptable_f1scores('f1.txt')


for i in hyperas_trialf1_list:
    cnn = models.load_model('%d-secstr_seq_denseconcat.h5' % i)
    pre_score = cnn.evaluate([testX1,testX2], testY, batch_size=32, verbose=0)
    color = colors[i]
    probas_ = cnn.predict([testX1,testX2])
    pred_score = probas_[:,1]
    true_class = testY[:,1]
    precision, recall, thresholds = precision_recall_curve(true_class, pred_score)
    plt.plot(recall, precision, lw=lw, color=color,label='hyperas trial %d (area = %0.3f)' % (i, auc(recall, precision)))
    y_real.append(true_class)
    y_proba.append(pred_score)
    precision_rnn.append(precision)
    recall_rnn.append(recall)
    threshold_rnn.append(thresholds)
    scores = np.where(pred_score > 0.5, 1, 0)
    tn, fp, fn, tp = confusion_matrix(true_class, scores).ravel()
    specificity =  tn / (tn+fp)
    sensitivity = tp/(tp+fn)
    precision_s = precision_score(true_class, scores)
    recall_s = recall_score(true_class, scores)
    f1_s = f1_score(true_class, scores)
    mcc = matthews_corrcoef(true_class,scores)
    fpr, tpr, thresholds = roc_curve(true_class, pred_score)
    roc_auc = auc(fpr, tpr)
    f1_all.append(f1_s)
    mcc_all.append(mcc)
    with open('imbalanced_results.txt', 'a') as file:
        file.write('#############hyperas trial '+str(i)+'#################'+'\n')
        file.write('test loss:' + str(pre_score[0]) + '\t' + 'test acc:' + str(pre_score[1]) + '\n' +
                   'precision:%.3f' % precision_score(y_true=true_class, y_pred=scores) + '\t'
                   'recall:%.3f' % recall_score(y_true=true_class, y_pred=scores) + '\t'
                   'F1:%.3f' % f1_score(y_true=true_class, y_pred=scores) + '\n'
                   'mcc:' + str(mcc) + '\t' + 'auc:' + str(roc_auc) + '\n' +
                   'sensitivity:'+str(sensitivity) + '\t' + 'specificity:' +str(specificity)+'\n')
        file.write('##############################'+'\n')
##Best model hyperas##
cnn = models.load_model('best_hyperas_model_secstr.h5')
probas_ = cnn.predict([testX1,testX2])
pred_score = probas_[:,1]
true_class = testY[:,1]
precision, recall, thresholds = precision_recall_curve(true_class, pred_score)
plt.plot(recall, precision, lw=lw, color='r',label='hyperas best model (area = %0.3f)' % (auc(recall, precision)))
y_real.append(true_class)
y_proba.append(pred_score)
precision_rnn.append(precision)
recall_rnn.append(recall)
threshold_rnn.append(thresholds)
scores = np.where(pred_score > 0.5, 1, 0)
tn, fp, fn, tp = confusion_matrix(true_class, scores).ravel()
specificity =  tn / (tn+fp)
sensitivity = tp/(tp+fn)
precision_s = precision_score(true_class, scores)
recall_s = recall_score(true_class, scores)
f1_s = f1_score(true_class, scores)
mcc = matthews_corrcoef(true_class,scores)
fpr, tpr, thresholds = roc_curve(true_class, pred_score)
roc_auc = auc(fpr, tpr)
f1_all.append(f1_s)
mcc_all.append(mcc)
with open('imbalanced_results.txt', 'a') as file:
        file.write('#############hyperas best model #################'+'\n')
        file.write('test loss:' + str(pre_score[0]) + '\t' + 'test acc:' + str(pre_score[1]) + '\n' +
                   'precision:%.3f' % precision_score(y_true=true_class, y_pred=scores) + '\t'
                   'recall:%.3f' % recall_score(y_true=true_class, y_pred=scores) + '\t'
                   'F1:%.3f' % f1_score(y_true=true_class, y_pred=scores) + '\n'
                   'mcc:' + str(mcc) + '\t' + 'auc:' + str(roc_auc) + '\n' +
                   'sensitivity:'+str(sensitivity) + '\t' + 'specificity:' +str(specificity)+'\n')
        file.write('##############################'+'\n')
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, thresholds = precision_recall_curve(y_real, y_proba)
precision_rnn.append(precision)
recall_rnn.append(recall)
threshold_rnn.append(thresholds)
plt.plot(recall, precision, lw=lw, color='g', linestyle='--',label='Mean (area = %0.3f)' % (auc(recall, precision)))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('RNN Precision Recall Curve')
plt.legend(loc="lower left")
plt.savefig('Hyperas_Precision-Recall_curve_imbalanced.png')
