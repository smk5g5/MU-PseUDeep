import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
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
##    from tensorflow.keras.callbacks import TensorBoard
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
from keras.optimizers import Nadam,Adam,RMSprop,SGD
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
import os
from sklearn import svm
from sklearn.manifold import TSNE
from matplotlib import offsetbox
from sklearn.metrics import accuracy_score, recall_score
import random
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperas.utils import eval_hyperopt_space
from keras.callbacks import TensorBoard
import globalvars
import cwt

#hyperas trial 26
##{'Dense': [1], 'Dropout': [0.9942741825824339], 'Dropout_1': [0.4084429796653032], 
##'Dropout_2': [0.5471399358933519], 'Dropout_3': [0.10926522237224338], 'batch_size': [0], 
##'choiceval': [1], 'l2': [0.011206678796947282], 'l2_1': [0.04663753230167181],
## 'l2_2': [0.032491576988669696], 'l2_3': [0.021775346680719416], 'l2_4': [0.06971582946481189], 
## 'l2_5': [0.010482932560304255], 'l2_6': [0.031598749327261345], 'l2_7': [0.008615890670714792]}

#hyperas best model

##{'Dense': 512, 'Dropout': 0.745150134528914, 'Dropout_1': 0.36965944352568686, 'Dropout_2': 0.66537
##16368558287, 'Dropout_3': 0.038045356303735206, 'batch_size': 32, 'choiceval': 'sgd', 'l2': 0.00109
##49235793883667, 'l2_1': 0.03405758144816304, 'l2_2': 0.03217477728270726, 'l2_3': 0.016089627620035
##51, 'l2_4': 0.08582474007227135, 'l2_5': 0.0014045290291504406, 'l2_6': 0.037289952982092284, 'l2_7
##': 0.01373965388919854}


def MCNN_best(trainX1,trainX2,trainY1,valX1,valX2,valY1,input_1,input_2, i,class_weights,t):
    if(t==0):
        print("####################################bootstrap iteration ",t,"#############fold iteration ",i,"\n")
        onehot_secstr = conv.Conv1D(5, 10, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.0010949235793883667), padding='valid', name='0_secstr')(input_1)
        onehot_secstr = Dropout(0.745150134528914)(onehot_secstr)
        onehot_secstr = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_secstr)
        onehot_secstr = core.Flatten()(onehot_secstr)
        onehot_secstr2 = conv.Conv1D(9, 4, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.03405758144816304), padding='valid', name='1_secstr')(input_1)
        onehot_secstr2 = Dropout(0.36965944352568686)(onehot_secstr2)
        onehot_secstr2 = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_secstr2)
        onehot_secstr2 = core.Flatten()(onehot_secstr2)
        output_onehot_sec = concatenate([onehot_secstr, onehot_secstr2], axis=-1)
        onehot_x = conv.Conv1D(5, 10, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.03217477728270726), padding='valid', name='0')(input_2)
        onehot_x = Dropout(0.6653716368558287)(onehot_x)
        onehot_x = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_x)
        onehot_x = core.Flatten()(onehot_x)
        onehot_x2 = conv.Conv1D(9, 4, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.01608962762003551), padding='valid', name='1')(input_2)
        onehot_x2 = Dropout(0.038045356303735206)(onehot_x2)
        onehot_x2 = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_x2)
        onehot_x2 = core.Flatten()(onehot_x2)
        output_onehot_seq = concatenate([onehot_x, onehot_x2], axis=-1)
        final_output = concatenate([output_onehot_sec, output_onehot_seq])
        dense_out = Dense(512, kernel_initializer='glorot_normal', activation='softplus', name='dense_concat')(final_output)
        out = Dense(2, activation="softmax", kernel_initializer='glorot_normal', name='6')(dense_out)
        ########## Set Net ##########
        cnn = Model(inputs=[input_1,input_2], outputs=out)
        cnn.load_weights('weightsfile_hyperasbest.h5')
        cnn.summary()
        adam = Adam(lr=0.08582474007227135)
        nadam = Nadam(lr=0.0014045290291504406)
        rmsprop = RMSprop(lr=0.037289952982092284)
        sgd = SGD(lr=0.01373965388919854)
        optim = sgd
##        choiceval = {{choice(['adam', 'sgd', 'rmsprop','nadam'])}}
##        if choiceval == 'adam':
##            optim = adam
##        elif choiceval == 'rmsprop':
##            optim = rmsprop
##        elif choiceval=='nadam':
##            optim = nadam
##        else:
##            optim = sgd
        cnn.compile(loss='binary_crossentropy', optimizer=optim, metrics=[keras.metrics.binary_accuracy])  # Nadam
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpointer = ModelCheckpoint(filepath='%d-secstr_seq_denseconcat_60perc_best.h5' % i, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
        fitHistory = cnn.fit([trainX1,trainX2], trainY1, batch_size=32, nb_epoch=500,validation_data=([valX1,valX2], valY1),callbacks=[checkpointer,early_stopping],class_weight=class_weights)
        myjson_file = "myhist_" +"dict_" + "secstr_seq_denseconcat_60perc_best_" +str(i)
        json.dump(fitHistory.history, open(myjson_file, 'w'))
        return cnn,fitHistory
    else:
        print("####################################bootstrap iteration ",t,"#############fold iteration ",i,"\n")
        cnn = models.load_model('%d-secstr_seq_denseconcat_60perc_best.h5' % i)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpointer = ModelCheckpoint(filepath='%d-secstr_seq_denseconcat_60perc_best.h5' % i, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
        fitHistory = cnn.fit([trainX1,trainX2], trainY1, batch_size=32, nb_epoch=500,validation_data=([valX1,valX2], valY1),class_weight=class_weights,callbacks=[checkpointer,early_stopping])
        myjson_file = "myhist_" +"dict_" + "secstr_seq_denseconcat_60perc_best_" +str(i)
        json.dump(fitHistory.history, open(myjson_file, 'a'))
        return cnn, fitHistory


def MCNN_26(trainX1,trainX2,trainY1,valX1,valX2,valY1,input_1,input_2, i,class_weights,t):
    if(t==0):
        print("####################################bootstrap iteration ",t,"#############fold iteration ",i,"\n")
        onehot_secstr = conv.Conv1D(5, 10, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.011206678796947282), padding='valid', name='0_secstr')(input_1)
        onehot_secstr = Dropout(0.9942741825824339)(onehot_secstr)
        onehot_secstr = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_secstr)
        onehot_secstr = core.Flatten()(onehot_secstr)
        onehot_secstr2 = conv.Conv1D(9, 4, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.04663753230167181), padding='valid', name='1_secstr')(input_1)
        onehot_secstr2 = Dropout(0.4084429796653032)(onehot_secstr2)
        onehot_secstr2 = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_secstr2)
        onehot_secstr2 = core.Flatten()(onehot_secstr2)
        output_onehot_sec = concatenate([onehot_secstr, onehot_secstr2], axis=-1)
        onehot_x = conv.Conv1D(5, 10, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.032491576988669696), padding='valid', name='0')(input_2)
        onehot_x = Dropout(0.5471399358933519)(onehot_x)
        onehot_x = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_x)
        onehot_x = core.Flatten()(onehot_x)
        onehot_x2 = conv.Conv1D(9, 4, kernel_initializer='glorot_normal',kernel_regularizer=l2(0.021775346680719416), padding='valid', name='1')(input_2)
        onehot_x2 = Dropout(0.10926522237224338)(onehot_x2)
        onehot_x2 = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_x2)
        onehot_x2 = core.Flatten()(onehot_x2)
        output_onehot_seq = concatenate([onehot_x, onehot_x2], axis=-1)
        final_output = concatenate([output_onehot_sec, output_onehot_seq])
        dense_out = Dense(30, kernel_initializer='glorot_normal', activation='softplus', name='dense_concat')(final_output)
        out = Dense(2, activation="softmax", kernel_initializer='glorot_normal', name='6')(dense_out)
        ########## Set Net ##########
        cnn = Model(inputs=[input_1,input_2], outputs=out)
        cnn.load_weights('weightsfile_hyperas26.h5')
        cnn.summary()
        adam = Adam(lr=0.06971582946481189)
        nadam = Nadam(lr=0.010482932560304255)
        rmsprop = RMSprop(lr=0.031598749327261345)
        sgd = SGD(lr=0.008615890670714792)
        optim = sgd
    ##        choiceval = {{choice(['adam', 'sgd', 'rmsprop','nadam'])}}
    ##        if choiceval == 'adam':
    ##            optim = adam
    ##        elif choiceval == 'rmsprop':
    ##            optim = rmsprop
    ##        elif choiceval=='nadam':
    ##            optim = nadam
    ##        else:
    ##            optim = sgd
        cnn.compile(loss='binary_crossentropy', optimizer=optim, metrics=[keras.metrics.binary_accuracy])  # Nadam
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpointer = ModelCheckpoint(filepath='%d-secstr_seq_denseconcat_60perc_trial26.h5' % i, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
        fitHistory = cnn.fit([trainX1,trainX2], trainY1, batch_size=32, nb_epoch=500,validation_data=([valX1,valX2], valY1),callbacks=[checkpointer,early_stopping],class_weight=class_weights)
        myjson_file = "myhist_" +"dict_" + "secstr_seq_denseconcat_60perc_trial26_" +str(i)
        json.dump(fitHistory.history, open(myjson_file, 'w'))
        return cnn,fitHistory
    else:
        print("####################################bootstrap iteration ",t,"#############fold iteration ",i,"\n")
        cnn = models.load_model('%d-secstr_seq_denseconcat_60perc_trial26.h5' % i)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpointer = ModelCheckpoint(filepath='%d-secstr_seq_denseconcat_60perc_trial26.h5' % i, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
        fitHistory = cnn.fit([trainX1,trainX2], trainY1, batch_size=32, nb_epoch=500,validation_data=([valX1,valX2], valY1),class_weight=class_weights,callbacks=[checkpointer,early_stopping])
        myjson_file = "myhist_" +"dict_" + "secstr_seq_denseconcat_60perc_trial26_" +str(i)
        json.dump(fitHistory.history, open(myjson_file, 'a'))
        return cnn, fitHistory
