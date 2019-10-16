from __future__ import print_function
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

#    class_weights = {0: 1,
#                1: 3}
#

"""
Created on Sat Feb 16 20:15:16 2019

@author: smk5g5
#Using Hyperas for hyperparameter optimization###
"""
import os
import pickle
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    from sklearn.utils import shuffle
    import numpy as np
    import pickle
    import pandas as pd
    from keras.layers import Input
    file1 = open("./imbalanced_testset/trainx1_0", 'rb') #sec str one hot matrix
    trainX1 = pickle.load(file1)
    file1 = open("./imbalanced_testset/trainx2_0", 'rb') #seq one hot matrix
    trainX2 = pickle.load(file1)
    file1 = open("./imbalanced_testset/trainy_0", 'rb') #labels
    trainY1 = pickle.load(file1)
    file1 = open("./imbalanced_testset/valx1_0", 'rb') 
    valX1 = pickle.load(file1)
    file1 = open("./imbalanced_testset/valx2_0", 'rb') 
    valX2 = pickle.load(file1)
    file1 = open("./imbalanced_testset/valy_0", 'rb') 
    valY1 = pickle.load(file1)
    file1 = open("./imbalanced_testset/testx1_0", 'rb')
    testX1 = pickle.load(file1)
    file1 = open("./imbalanced_testset/testx2_0", 'rb')
    testX2 = pickle.load(file1)
    file1 = open("./imbalanced_testset/testy_0", 'rb')
    testY = pickle.load(file1)
    trainX1, trainX2, trainY1 = shuffle(trainX1, trainX2, trainY1)
    valX1,valX2, valY1 = shuffle(valX1,valX2, valY1)
    testX1,testX2,testY = shuffle(testX1,testX2,testY)
    return trainX1, trainX2, trainY1, valX1,valX2, valY1,testX1,testX2,testY

##def step_decay(epoch):
##    initial_lrate = 0.1
##    drop=0.5
##    epochs_drop = 10.0
##    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
##    return lrate


def f1_score(true_class,pred_score):
    from sklearn.metrics import precision_score, recall_score, f1_score
    for index in range(len(pred_score)):
        if pred_score[index] > 0.5:
            pred_score[index] = 1
        else:
            pred_score[index] = 0
    myf1 = f1_score(y_true=true_class, y_pred=pred_score)
    return myf1


def MCNN(trainX1,trainX2,trainY1,valX1,valX2,valY1,testX1,testX2,testY):
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
##    from tensorflow.keras.callbacks import TensorBoard
    row1,col1 = trainX1[0].shape
    input_1 = Input(shape=(row1,col1))
    row2,col2 = trainX2[0].shape
    input_2 = Input(shape=(row2,col2))
    NAME = "combined_secstr_seq_CNN_model_emboss-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    onehot_secstr = conv.Conv1D(5, 10, kernel_initializer='glorot_normal',kernel_regularizer=l2({{uniform(0.0001, 0.1)}}), padding='valid', name='0_secstr')(input_1)
    onehot_secstr = Dropout({{uniform(0, 1)}})(onehot_secstr)
    onehot_secstr = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_secstr)
    onehot_secstr = core.Flatten()(onehot_secstr)
    onehot_secstr2 = conv.Conv1D(9, 4, kernel_initializer='glorot_normal',kernel_regularizer=l2({{uniform(0.0001, 0.1)}}), padding='valid', name='1_secstr')(input_1)
    onehot_secstr2 = Dropout({{uniform(0, 1)}})(onehot_secstr2)
    onehot_secstr2 = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_secstr2)
    onehot_secstr2 = core.Flatten()(onehot_secstr2)
    output_onehot_sec = concatenate([onehot_secstr, onehot_secstr2], axis=-1)
    onehot_x = conv.Conv1D(5, 10, kernel_initializer='glorot_normal',kernel_regularizer=l2({{uniform(0.0001, 0.1)}}), padding='valid', name='0')(input_2)
    onehot_x = Dropout({{uniform(0, 1)}})(onehot_x)
    onehot_x = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_x)
    onehot_x = core.Flatten()(onehot_x)
    onehot_x2 = conv.Conv1D(9, 4, kernel_initializer='glorot_normal',kernel_regularizer=l2({{uniform(0.0001, 0.1)}}), padding='valid', name='1')(input_2)
    onehot_x2 = Dropout({{uniform(0, 1)}})(onehot_x2)
    onehot_x2 = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None,alpha_constraint=None, shared_axes=None)(onehot_x2)
    onehot_x2 = core.Flatten()(onehot_x2)
    output_onehot_seq = concatenate([onehot_x, onehot_x2], axis=-1)
    final_output = concatenate([output_onehot_sec, output_onehot_seq])
    dense_out = Dense({{choice([20,30,50,60,64,70,80,90,100, 128, 256, 512, 1024])}}, kernel_initializer='glorot_normal', activation='softplus', name='dense_concat')(final_output)
    out = Dense(2, activation="softmax", kernel_initializer='glorot_normal', name='6')(dense_out)
    ########## Set Net ##########
    cnn = Model(inputs=[input_1,input_2], outputs=out)
    cnn.summary()
    adam = Adam(lr={{uniform(0.0001, 0.1)}})
    nadam = Nadam(lr={{uniform(0.0001, 0.1)}})
    rmsprop = RMSprop(lr={{uniform(0.0001, 0.1)}})
    sgd = SGD(lr={{uniform(0.0001, 0.1)}})
    choiceval = {{choice(['adam', 'sgd', 'rmsprop','nadam'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    elif choiceval=='nadam':
        optim = nadam
    else:
        optim = sgd
    globalvars.globalVar += 1
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
    cnn.compile(loss='binary_crossentropy', optimizer=optim, metrics=[keras.metrics.binary_accuracy])  # Nadam
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    checkpointer = ModelCheckpoint(filepath='%d-secstr_seq_denseconcat.h5' % globalvars.globalVar, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
    fitHistory = cnn.fit([trainX1,trainX2], trainY1, batch_size={{choice([32,64,128,256,512])}}, nb_epoch=500,validation_data=([valX1,valX2], valY1),callbacks=[checkpointer,early_stopping,tensorboard],class_weight=cwt.class_weights)
    myjson_file = "myhist_" +"_dict" + "_hyperas_model_trial_" +str(globalvars.globalVar)
    json.dump(fitHistory.history, open(myjson_file, 'w'))
    score, acc = cnn.evaluate([valX1, valX2], valY1, batch_size=32)
    pred_proba = cnn.predict([valX1,valX2], batch_size=32)
    pred_score = pred_proba[:, 1]
    true_class = valY1[:, 1]
    f1_sc = f1_score(true_class,pred_score)
    print('F1 score:', f1_sc)
    print('Test score:', score)
    print('accuracy:', acc)
    return {'loss': -f1_sc, 'status': STATUS_OK, 'model': cnn}

if __name__ == '__main__':
    trials=Trials()
    best_run, best_model,space = optim.minimize(model=MCNN,
                                          data=data,
                                          functions=[f1_score],
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=trials,
                                          eval_space=True,
                                        return_space=True)
    trainX1,trainX2,trainY1,valX1,valX2,valY1,testX1,testX2,testY = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate([valX1, valX2], valY1, verbose=0))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save('best_hyperas_model_secstr.h5')
    hyperas_dict = dict()
    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        hyperas_dict[t] = vals
        print("Trial %s vals: %s" % (t, vals))  # <-- indices
        print(eval_hyperopt_space(space, vals))  # <-- values
##from hyperas.utils import eval_hyperopt_space
##print(eval_hyperopt_space(space, vals))
pickle.dump(hyperas_dict, open( "./hyperas_dict_100trials", "wb" ))        
