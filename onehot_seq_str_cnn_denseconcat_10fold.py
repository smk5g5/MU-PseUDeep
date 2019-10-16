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

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

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



def MCNN(trainX1,trainX2,trainY1,valX1,valX2,valY1,input_1,input_2, i,class_weights):
    onehot_secstr = conv.Conv1D(5, 10, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.04), padding='valid', name='0_secstr')(input_1)
    onehot_secstr = Dropout(0.6)(onehot_secstr)
    onehot_secstr = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_secstr)
    onehot_secstr = core.Flatten()(onehot_secstr)
    onehot_secstr2 = conv.Conv1D(9, 4, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.02), padding='valid', name='1_secstr')(input_1)
    onehot_secstr2 = Dropout(0.4)(onehot_secstr2)
    onehot_secstr2 = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_secstr2)
    onehot_secstr2 = core.Flatten()(onehot_secstr2)
    output_onehot_sec = concatenate([onehot_secstr, onehot_secstr2], axis=-1)
    onehot_x = conv.Conv1D(5, 10, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.04), padding='valid', name='0')(input_2)
    onehot_x = Dropout(0.6)(onehot_x)
    onehot_x = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_x)
    onehot_x = core.Flatten()(onehot_x)
    onehot_x2 = conv.Conv1D(9, 4, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.02), padding='valid', name='1')(input_2)
    onehot_x2 = Dropout(0.4)(onehot_x2)
    onehot_x2 = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_x2)
    onehot_x2 = core.Flatten()(onehot_x2)
    output_onehot_seq = concatenate([onehot_x, onehot_x2], axis=-1)
    final_output = concatenate([output_onehot_sec, output_onehot_seq])
    dense_out = Dense(100, kernel_initializer='glorot_normal', activation='softplus', name='dense_concat')(final_output)
    out = Dense(2, activation="softmax", kernel_initializer='glorot_normal', name='6')(dense_out)
    ########## Set Net ##########
    cnn = Model(inputs=[input_1,input_2], outputs=out)
    cnn.summary()
    nadam = Nadam(lr=0.001)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
    cnn.compile(loss='binary_crossentropy', optimizer=nadam, metrics=[keras.metrics.binary_accuracy])  # Nadam
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    checkpointer = ModelCheckpoint(filepath='%d-secstr_seq_denseconcat_60perc.h5' % i, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
    fitHistory = cnn.fit([trainX1,trainX2], trainY1, batch_size=256, nb_epoch=500,validation_data=([valX1,valX2], valY1),class_weight=class_weights,callbacks=[checkpointer,early_stopping])
    history_dict = fitHistory.history
    myjson_file = "myhist_" +"dict_" + "secstr_seq_denseconcat_60perc_" +str(i)
    json.dump(history_dict, open(myjson_file, 'w'))
    return cnn, fitHistory

#trained_model, testX1,testX2, testY, iteration, 'combined_model'
def test_in(testX1,testX2, testY, i, argument):
    """
    You can put the val data as a reference adjustment model,
    or you can put the test data evaluation model.
    """
    cnn = models.load_model('%d-secstr_seq_denseconcat_60perc.h5' % i)
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
    seq_test,secstr_test,label_test = get_all_seq(test)
    testX2,testY = onehotkey(seq_test,label_test)
    testX1,testY = onehotkey_sec(secstr_test,label_test)
    testY = np_utils.to_categorical(testY,2)
    testY = testY.reshape(-1,2)
    row1,col1 = testX1[0].shape
    row2,col2 = testX2[0].shape
    testX1.shape = (testX1.shape[0],row1,col1)
    testX2.shape = (testX2.shape[0],row2,col2)
    testx1_out = "./testx1_"+str(i)
    testx2_out = "./testx2_"+str(i)
    testy_out = "./testy_"+str(i)
    pickle.dump(testX1, open(testx1_out, "wb" ))
    pickle.dump(testX2, open(testx2_out, "wb" ))
    pickle.dump(testY, open(testy_out, "wb" ))
    pre_score = test_in(testX1,testX2, testY, iteration, 'combined_model')
    print('test loss:', pre_score[0],'test acc:', pre_score[1])

##infile = "emboss_60perc_test_"+str(i)+".txt"
##emboss_60perc_train_8.txt
for i in range(10):
    train_file = "emboss_60perc_train_"+str(i)+".txt"
    valid_file = "emboss_60perc_validation_"+str(i)+".txt"
    test_file = "emboss_60perc_testing_"+str(i)+".txt"
    train = read_composite_data(train_file)
    train = train.sample(frac=1)
    valid = read_composite_data(valid_file)
    valid = valid.sample(frac=1)
    test = read_composite_data(test_file)
    test = test.sample(frac=1)
    ########################### validation stratification#########
    pos_train = train.loc[train['Label']==1]
    neg_train = train.loc[train['Label']==0]
    #ratio of negative to pos for train data
    neg_pos_ratio = neg_train.shape[0]/len(pos_train)
    #all positive data
    pos_valid = valid.loc[valid['Label']==1]
    #all negative data
    neg_valid = valid.loc[valid['Label']==0]
    #size of negative data we want for validation set
    neg_size = int(pos_valid.shape[0]*neg_pos_ratio)
    rand_negsel = neg_valid.sample(n=neg_size)
    valid_new = pd.concat([pos_valid,rand_negsel])
    valid_new = valid_new.sample(frac=1)
    #expected size of validation data
    valid_size = int(0.2 * len(train))
    valid_tr,valid_te = train_test_split(valid_new,train_size=valid_size,stratify=valid_new['Label'])
##    test_new =  pd.concat([valid_te,test])
##    test_new = test_new.sample(frac=1)
    seq,secstr,label = get_all_seq(train)
    seq_val,secstr_val,label_val = get_all_seq(valid_tr)
    trainX2, trainY = onehotkey(seq, label)
    trainX1, trainY = onehotkey_sec(secstr, label)
##    class_weight_list = compute_class_weight('balanced', np.unique(trainY), trainY)
##    class_weights = dict(zip(np.unique(trainY), class_weight_list))
    class_weights = {0: 1,
                1: 3}
    trainY = np_utils.to_categorical(trainY,2)
    row1,col1 = trainX1[0].shape
    trainX1.shape = (trainX1.shape[0], row1,col1)
    input_1 = Input(shape=(row1,col1))
    row2,col2 = trainX2[0].shape
    trainX2.shape = (trainX2.shape[0], row2,col2)
    input_2 = Input(shape=(row2,col2))
    trainY = trainY.reshape(-1,2)
    ValX2, ValY = onehotkey(seq_val, label_val)
    ValX1,ValY = onehotkey_sec(secstr_val, label_val)
    ValY = np_utils.to_categorical(ValY,2)
    ValY = ValY.reshape(-1,2)
    cnn, fitHistory = MCNN(trainX1,trainX2,trainY,ValX1,ValX2,ValY,input_1,input_2, i,class_weights)
    loss1, acc1, loss2, acc2 = print_loss(fitHistory, i, 'sec_str_plus_seq_cnn')
    print('train loss:', loss1,  'train acc:', acc1,'val loss:', loss2, 'val acc:', acc2)
    trainx1_out = "./trainx1_"+str(i)
    trainx2_out = "./trainx2_"+str(i)
    trainy_out = "./trainy_"+str(i)
    valx1_out = "./valx1_"+str(i)
    valx2_out = "./valx2_"+str(i)
    valy_out = "./valy_"+str(i)
    pickle.dump(trainX1, open(trainx1_out, "wb" ))
    pickle.dump(trainX2, open(trainx2_out, "wb" ))
    pickle.dump(trainY, open(trainy_out, "wb" ))
    pickle.dump(ValX1, open(valx1_out, "wb" ))
    pickle.dump(ValX2, open(valx2_out, "wb" ))
    pickle.dump(ValY, open(valy_out, "wb" ))
    run_test_onehot(test,i)
