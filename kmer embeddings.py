# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:43:12 2019

@author: smk5g5
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
import math
import numpy as np

import os
import pandas as pd
from collections import defaultdict
from Bio import SeqIO
from nltk import bigrams
from nltk import trigrams
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler, History

from keras.layers import Dropout
from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras import regularizers
from keras.layers import GaussianNoise
from keras.layers import Activation
from keras.callbacks import LearningRateScheduler, EarlyStopping

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import regularizers
#from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, GRU
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.models import load_model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import regularizers
#from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, GRU
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.models import load_model
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer

import dask.dataframe as dd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from scipy import interp
from itertools import cycle
import sys
import collections
import nltk
from collections import defaultdict
from nltk import bigrams
from nltk import trigrams
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import pandas as pad
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, stream = sys.stdout)
import numpy as np
import tensorflow as tf
import random as rn
from sklearn.utils.class_weight import compute_class_weight

def concatenate_tup(mytup):
    concat_str=''
    for i in mytup:
        concat_str = concat_str+i
    return(concat_str)

class MySentences_kmer(object):
    
    def __init__(self):
        pass
    
    def __iter__(self):
        for index, record in enumerate(SeqIO.parse('./final_m6Arevcomp_51.fa', 'fasta')):
            for loop_num in range(0, 3):
                word_list = []
                #print record.seq
                tri_tokens = nltk.everygrams(record.seq,min_len=3,max_len=8)
                for index1, item in enumerate(tri_tokens):
                    if index1 % 3 == loop_num:
                        tri_str = concatenate_tup(item)
                        #print tri_str,
                        word_list.append(tri_str)
                #print
                yield word_list
                
#try with window sizes 3,5 and 7
#try with various embedding dimension sizes : 128,256,512,1024
def make_word_embeddings(seq_object,embedding_dim,window_size):
    sg = 1
    model = gensim.models.Word2Vec(seq_object, min_count=0, size=embedding_dim, window=window_size, sg = sg, workers = 10)
    return model


#kmerseqs = MySentences_kmer()
#w2vec_model = make_word_embeddings(kmerseqs,300,7)


def read_composite_data(file):
    df = pd.read_csv(file,sep="\t",names=["Sequence","Chr_loc","Label"])
    return(df)

def get_all_seq(df):
    seq = []
    label = []
    for indexs in df.index:
        seq.append(df.loc[indexs].values[0])
        label.append(df.loc[indexs].values[2])
    return seq, label




import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
seed = 42
np.random.seed(seed)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def concatenate_tup(mytup):
    concat_str=''
    for i in mytup:
        concat_str = concat_str+i
    return(concat_str)


def select_max_seqlen(seqlist):
    max_list = []
    for i in seqlist:
        temp_list = i.split(" ")
        uniq_list = list(set(temp_list[1:-1]))
        max_list.append(len(uniq_list))
    max_list.sort()
    return max_list[-1]

def prepare_seqs(rnaseq_list):
    texts = []
    for i in rnaseq_list:
        tri_tokens = nltk.everygrams(i,min_len=3,max_len=8)
        temp_str = ""
        count = 0
        for item in ((tri_tokens)):
            variable_kmer = concatenate_tup(item)
            if(count==0):
                temp_str = variable_kmer
                count=count+1
            else:
                temp_str = temp_str + " " + variable_kmer
        texts.append(temp_str)
    return texts

def tokenize_sequences(sequence_kmers,MAX_NB_WORDS):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS) #MAX_NB_WORDS = 10334
    tokenizer.fit_on_texts(sequence_kmers)
    sequences = tokenizer.texts_to_sequences(sequence_kmers)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    count = len(word_index)
    return word_index,sequences

    
    
def make_embedding_matrix(MAX_NB_WORDS,EMBEDDING_DIM,word_index,model):
    embedding_matrix = np.zeros((MAX_NB_WORDS+1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = model[word.upper()]
        if i >= MAX_NB_WORDS:
            continue
        elif word.upper() in model.wv.vocab:
            embedding_vector = model[word.upper()]
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def make_embedding_layer(MAX_NB_WORDS,EMBEDDING_DIM,embedding_matrix,MAX_SEQUENCE_LENGTH):
    embedding_layer = Embedding(MAX_NB_WORDS+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
    return embedding_layer

def create_model_no_pretrain_bidirec():
    print('New model for Nm2Ome')


    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), )
    embedded_sequences = embedding_layer(sequence_input)
    # before recurrent_dropout was 0.5
    x = Bidirectional(GRU(32, dropout=0.5, recurrent_dropout=0.1, return_sequences = True))(embedded_sequences) 
    x = Bidirectional(GRU(32, dropout=0.7,recurrent_dropout=0.1))(x)
#    x = Bidirectional(GRU(16, dropout=0.7,recurrent_dropout=0.1))(x)
#    x = Bidirectional(GRU(16, dropout=0.5,recurrent_dropout=0.1))(x)
#    x = Bidirectional(GRU(16, dropout=0.5,recurrent_dropout=0.1))(x)
#    x = MaxPooling1D(35)(x)  # global max pooling
#    x = GRU(32, dropout=0.1, recurrent_dropout=0.5)(x)
#    x = Flatten()(x)
#    x = Dense(1028, activation='relu')(x)
    # x = Dense(1028, activation='relu')(x)
    # x = Dense(1028, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.6)(x)
    preds = Dense(1, activation='sigmoid')(x)

    new_model = Model(sequence_input, preds)
    new_model.compile(loss='binary_crossentropy',
                  optimizer= 'adam',
                  metrics=['acc'])
    
    #new_model.layers[2].set_weights((model.layers[2].get_weights()))
    return new_model
    
#try with window sizes 3,5 and 7
#try with various embedding dimension sizes : 128,256,512,1024
def test_in(testX, testY, i, argument):
    cnn = load_model('%d-onehot.h5' % i)
    #  ############### test ##########################
    pre_score = cnn.evaluate(testX, testY, batch_size=1024, verbose=0)
    #  ######### Print Precision and Recall ##########
    pred_proba = cnn.predict(testX, batch_size=1024)
    pred_score = np.where(pred_proba > 0.5, 1, 0)
    precision, recall, _ = precision_recall_curve(testY, pred_score)
    average_precision = average_precision_score(testY, pred_score)
    fpr, tpr, thresholds = roc_curve(testY, pred_score)
    roc_auc = auc(fpr, tpr)
    mcc = matthews_corrcoef(testY, pred_score)
    print('precision:%.3f' % precision_score(y_true=testY, y_pred=pred_score))
    print('recall:%.3f' % recall_score(y_true=testY, y_pred=pred_score))
    print('F1:%.3f' % f1_score(y_true=testY, y_pred=pred_score))
    plt.figure()
    plt.step(recall, precision, color='navy', where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('./plt/' + str(argument) + 'Precision-Recall%d.png' % i)
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
                   'precision:%.3f' % precision_score(y_true=testY, y_pred=pred_score) + '\t'
                   'recall:%.3f' % recall_score(y_true=testY, y_pred=pred_score) + '\t'
                   'F1:%.3f' % f1_score(y_true=testY, y_pred=pred_score) + '\n'
                   'mcc:' + str(mcc) + '\t' + 'auc:' + str(roc_auc) + '\n')
    return pre_score

def print_loss(fitHistory, i, argument):
    plt.figure()
    plt.plot(fitHistory.history['loss'][:-20])  # patience in earlystopping.
    plt.plot(fitHistory.history['val_loss'][:-20])
    plt.title('LOSS:times %d' % i)
    plt.ylim([0.35, 1.0])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./plt/' + str(argument) + 'loss%d.png' % i)
    loss1 = fitHistory.history['loss'][-21:-20]
    acc1 = fitHistory.history['binary_accuracy'][-21:-20]
    loss2 = fitHistory.history['val_loss'][-21:-20]
    acc2 = fitHistory.history['val_binary_accuracy'][-21:-20]
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


training_file = sys.argv[1]
test_file = sys.argv[2]
#w2vec_model.save('word2vec_model_m6A_windowsize7_dim300')
w2vec_model = gensim.models.Word2Vec.load('word2vec_model_m6A_windowsize7_dim300')
train = read_composite_data(training_file)
train_seq,train_label = get_all_seq(train)
test = read_composite_data(test_file)
test_seq,test_label = get_all_seq(test)
train_label = np.array(train_label)
test_label = np.array(test_label) 

class_weight_list = compute_class_weight('balanced', np.unique(train_label), train_label)
class_weights = dict(zip(np.unique(train_label), class_weight_list))
i = 0

train_kmers = prepare_seqs(train_seq)
test_kmers = prepare_seqs(test_seq)

MAX_NB_WORDS = len(w2vec_model.wv.vocab)
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = select_max_seqlen(train_kmers)


train_wordindex,train_sequences = tokenize_sequences(train_kmers,MAX_NB_WORDS)
train_embedding = make_embedding_matrix(MAX_NB_WORDS,EMBEDDING_DIM,train_wordindex,w2vec_model)
embedding_layer = make_embedding_layer(MAX_NB_WORDS,EMBEDDING_DIM,train_embedding,MAX_SEQUENCE_LENGTH)

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(train_data.shape)
checkpointer = ModelCheckpoint(filepath='%d-bidirectional_rnn.h5' % i, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
new_model = None
new_model = create_model_no_pretrain_bidirec()
rnn_history = new_model.fit(train_data, train_label, validation_split=0.2,epochs=500, batch_size=1024,callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=50),checkpointer],class_weight=class_weights)
loss1, acc1, loss2, acc2 = print_loss(rnn_history, 0, 'assorted_kmers_rnn')
test_wordindex,test_sequences = tokenize_sequences(test_kmers,MAX_NB_WORDS)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_in(test_data,test_label,0,'assorted_kmers_rnn')

def create_model_CNN():
    print('New model for Nm2Ome')


    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    print(sequence_input)
    exit()
    embedded_sequences = embedding_layer(sequence_input)
    # before recurrent_dropout was 0.5
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
    return model

checkpointer = ModelCheckpoint(filepath='%d-conv1d.h5' % i, verbose=1,save_best_only=True, monitor='val_loss', mode='min')
cnnmodel = None
cnnmodel = create_model_CNN()
cnn_history = cnnmodel.fit(train_data, train_label, validation_split=0.2,epochs=500, batch_size=1024,callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=50),checkpointer],class_weight=class_weights)
loss1, acc1, loss2, acc2 = print_loss(cnn_history, 0, 'assorted_kmers_rnn')
test_in(test_data,test_label,0,'assorted_kmers_cnn')
