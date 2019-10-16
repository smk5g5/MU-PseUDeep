# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:45:49 2018

@author: smk5g5
"""

# function from data file i get to data can input CNN
import os
import pickle
import random
import numpy as np
from pandas import DataFrame
#from keras.layers import Input
#import keras.utils.np_utils as kutils
from sklearn.model_selection import train_test_split


def get_gene_name(filename):
    """
    read file download from RMBase and get all Gene name in it.
    :param filename:name of the file you want to process
    :return: list gene name in file
    """
    with open(filename) as file:
        gene_name = []
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('#'):
                i += 1
            else:
                line = lines[i].split('\t')
                names = line[11].split(',')
                for name in names:
                    gene_name.append(name)
    gene_name = list(set(gene_name))
    del lines
    return gene_name


def gene_id(gene_name, filename):
    """
    use gene name get gene id
    :param gene_name: gene name list
    :param filename:The file that gets the gene name.(gene code file (gtf))
    :return:gene id list
    """
    with open(filename) as file:
        geneid = []
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('#'):
                i += 1
            else:
                line = lines[i].split('\t')
                annotation = line[8].replace(' ', '').split(';')
                for a in range(len(annotation)):
                    item = annotation[a]
                    if item.startswith('gene_name'):
                        name = item[9:].replace('"', '')
                        if name in gene_name:
                            for b in range(len(annotation)):
                                if annotation[b].startswith('gene_id'):
                                    id = annotation[b][7:].replace('"', '')
                                    geneid.append(id)
    return geneid

def get_gene_id(filepath, gene_name):
    """
    use geneid function deal file in file path
    :param filepath:the name in file path (list)
    :param gene_name: gene name list
    :return:id list
    """
    filenames = findfile(filepath)
    ids = []
    for filename in filenames:
        if filename.endswith('gtf'):
            geneid = gene_id(gene_name, filepath + '/' + filename)
            ids += geneid
    ids = list(set(ids))
    return ids

def findfile(filepath):
    """
    get file name in file path
    :param filepath: the path
    :return: file name list
    """
    filenames = []
    filepath = os.path.normcase(filepath)
    files = os.listdir(filepath)
    for filename in files:
        filenames.append(filename)
    return filenames


def fa_to_list(fileloc):
    """
    Chenge fasta to two lists
    :param fileloc:the path fasta file in
    :return:list_id gere id of the seq,list_seq seq
    """
    list_id = []
    list_seq = []
    with open(fileloc) as f:
        ll = []
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('>'):
                if ll != []:
                    new_seq = ''.join(ll)
                    list_seq.append(new_seq)
                xx = lines[i].split('|')
                list_id.append(xx[1])
                ll = []
                i += 1
            else:
                line = str(lines[i]).strip('\n')
                ll.append(line)
        new_last = ''.join(ll)
        list_seq.append(new_last)
    return list_id, list_seq


def get_gene_seq(filepath, gene_id, fileout):
    """
    if seq name in gene_id write it in a pickle
    :param filepath:the path fasta file in
    :param gene_id:list_id id
    :return:NONE
    """
    filenames = findfile(filepath)
    data = []
    for filename in filenames:
        if filename.endswith('fa'):
            list_id, list_seq = fa_to_list(filepath + '/' + filename)
            for a in range(len(list_id)):
                for id in gene_id:
                    if list_id[a] == id:
                        data.append(list_seq[a].strip('\n'))
    file = open(fileout, 'wb')
    pickle.dump(data, file)
    file.close()


def subseq_com(seq, id, num):
    """
    cut seq with 'N'
    :param seq:
    :param id:
    :param num:
    :return: seq (str)
    """
    win = num
    sub_seq = ''
    if (id-win) < 0 and (id + win) > len(seq):
        for i in range(win-id):
            sub_seq += 'N'
        for i in range(0, len(seq)):
            sub_seq += seq[i]
        for i in range(len(seq), id+win+1):
            sub_seq += 'N'
    elif (id-win) < 0 and (id+win+1) <= len(seq):
        for i in range(win-id):
            sub_seq += 'N'
        for i in range(0, id+win+1):
            sub_seq += seq[i]
    elif (id-win) >= 0 and (id+win+1) > len(seq):
        for i in range(id-win, len(seq)):
            sub_seq += seq[i]
        for i in range(len(seq), id+win+1):
            sub_seq += 'N'
    elif (id-win) >= 0 and (id+win+1) <= len(seq):
        for i in range(id-win, id+win+1):
            sub_seq += seq[i]
    return sub_seq


def subseq(seq, id, num):
    """
    cut seq with out 'N'
    :param seq:
    :param id:
    :param num:
    :return: seq(str)
    """
    win = num
    sub_seq = ''
    if (id-win) >= 0 and (id+win+1) <= len(seq):
        for i in range(id-win, id+win+1):
            sub_seq += seq[i]
    return sub_seq

def cut_seq(filein, fileout):
    """
    
    :param filein:
    :param fileout:
    :return:
    """
    file_db2 = open(fileout, 'w')
    i = 1
    with open(filein, 'rb') as file:
        seq_list = pickle.load(file)
        for seq in seq_list:
            for a in range(len(seq)):
                if seq[a] == 'A':
                    after_cut = subseq(seq, a, 20)
                    if after_cut != '':
                        file_db2.write('>N%d\n' % i)
                        file_db2.write(after_cut)
                        file_db2.write('\n')
                        i += 1
    file_db2.close()

def pos_to_fa(filein, fileout):
    """
    
    :param filein:
    :param fileout:
    :return:
    """
    file_db1 = open(fileout, 'w')
    a = 1
    with open(filein) as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('#'):
                i += 1
            else:
                line = lines[i].split('\t')
                other = line[-2].replace('\n', '')
                file_db1.write('>P%d\n' % a)
                file_db1.write(other)
                file_db1.write('\n')
                a += 1
    file_db1.close()

def deduplication(file_in_pos, file_in_all, fileout):
    """
    
    :param filein_pos:
    :param filein_all:
    :param fileout:
    :return:
    """
    pos_seq = []
    with open(file_in_pos) as file_1:
        lines_1 = file_1.readlines()
        for i in range(len(lines_1)):
            if lines_1[i].startswith('>'):
                continue
            else:
                pos_seq.append(lines_1[i])
    ret = []
    with open(file_in_all) as file_2:
        lines = file_2.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('>'):
                continue
            else:
                if lines[i] not in pos_seq:
                    ret.append(lines[i])
    with open(fileout) as file:
        a = 1
        for i in ret:
            file.write('>N%d\n' % a)
            file.write(i)
            file.write('\n')

def fa_to_df(filename):
    """
    
    :param filename:
    :return:
    """
    name = []
    seq = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('>'):
                name.append(lines[i])
            else:
                seq.append(lines[i])
    df = DataFrame({'name': name, 'seq': seq})
    return df

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


def df_to_fa(df, filename):
    """
    
    :param df:
    :param filename:
    :return:
    """
    data_final = np.array(df)
    data_list = data_final.tolist()
    file = open(filename, 'w')
    for i in data_list:
        file.write(i[0])
        file.write(i[1])
    file.close()

def split_data(data_pos, data_neg, trainfile, testfile, rate):
    """
    
    :param data_pos:
    :param data_neg:
    :param trainfile:
    :param testfile:
    :param rate:
    :return:
    """
    df_neg = fa_to_df(data_pos)
    df_pos = fa_to_df(data_neg)
    df_neg_train, df_neg_test = per_split(df_neg, rate)
    df_pos_train, df_pos_test = per_split(df_pos, rate)
    df_train = df_neg_train.append(df_pos_train)
    df_test = df_neg_test.append(df_pos_test)
    df_train = df_train.sample(frac=1)
    df_test = df_test.sample(frac=1)
    df_to_fa(df_train, trainfile)
    df_to_fa(df_test, testfile)

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

def AAindexVector(shuffled_SEQ, shuffled_TAG):
    """add T"""
    tag = np.array(shuffled_TAG)
    letterDict = {}
    letterDict["GG"] = [-0.01, -1.78, 3.32, 0.30, 12.10, 32.00, -11.10, -12.20, -29.70, -3.26, 0.17]
    letterDict["GA"] = [0.07, -1.70, 3.38, 1.30, 9.40, 32.00, -14.20, -13.30, -35.50, -2.35, 0.10]
    letterDict["GC"] = [0.07,-1.39,3.22,0.00,6.10,35.00,-16.90,-14.20,-34.90,-3.42,0.26]
    letterDict["GU"] = [0.23, -1.43, 3.24, 0.80, 4.80, 32.00, -13.80, -10.20, -26.20, -2.24, 0.27]
    letterDict["AG"] = [-0.04, -1.50, 3.30, 0.50, 8.50, 30.00, -14.00, -7.60, -19.20, -2.08, 0.08]
    letterDict["AA"] = [-0.08, -1.27, 3.18, -0.80, 7.00, 31.00, -13.70, -6.60, -18.40, -0.93, 0.04]
    letterDict["AC"] = [0.23, -1.43, 3.24, 0.80, 4.80, 32.00, -13.80, -10.20, -26.20, -2.24, 0.14]
    letterDict["AU"] = [-0.06, -1.36, 3.24, 1.10, 7.10, 33.00, -15.40, -5.70, -15.50, -1.10, 0.14]
    letterDict["CG"] = [0.30, -1.89, 3.30, -0.10, 12.10, 27.00, -15.60, -8.00, -19.40, -2.36, 0.35]
    letterDict["CA"] = [0.11, -1.46, 3.09, 1.00, 9.90, 31.00, -14.40, -10.50, -27.80, -2.11, 0.21]
    letterDict["CC"] = [-0.01, -1.78, 3.32, 0.30, 8.70, 32.00, -11.10, -12.20, -29.70, -3.26, 0.49]
    letterDict["CU"] = [-0.04, -1.50, 3.30, 0.50, 8.50, 30.00, -14.00, -7.60, -19.20, -2.08, 0.52]
    letterDict["UG"] = [0.11, -1.46, 3.09, 1.00, 9.90, 31.00, -14.40, -7.60, -19.20, -2.11, 0.34]
    letterDict["UA"] = [-0.02, -1.45, 3.26, -0.20, 10.70, 32.00, -16.00, -8.10, -22.60, -1.33, 0.21]
    letterDict["UC"] = [0.07, -1.70, 3.38, 1.30, 9.40, 32.00, -14.20, -10.20, -26.20, -2.35, 0.48]
    letterDict["UU"] = [-0.08, -1.27, 3.18, -0.80, 7.00, 31.00, -13.70, -6.60, -18.40, -0.93, 0.44]
    AACategoryLen = 11
    probMatr = np.zeros((len(shuffled_SEQ), len(shuffled_SEQ[0])-1, AACategoryLen))
    sampleNo = 0
    for sequence in shuffled_SEQ:
        list = []
        for a in range(len(sequence)):
            try:
                b = sequence[a] + sequence[a+1]
                list.append(b)
            except:
                continue
        AANo = 0
        for AA in list:
            try:
                if AA in letterDict:
                    probMatr[sampleNo][AANo] = letterDict[AA]
                else:
                    continue
                AANo += 1
            except:
                AANo += 1
        sampleNo += 1
        return probMatr, tag

def getseq(df):
    """
    
    :param df:
    :return:
    """
    seq = []
    label = []
    for indexs in df.index:
        seq.append(df.loc[indexs].values[1])
        name = df.loc[indexs].values[0]
        if name.replace(">", "").rstrip().startswith("N"):
            label.append(0)
        else:
            label.append(1)
    for i in range(len(seq)):
        seq[i] = seq[i].strip('\n')
    return seq, label


def save_data_train(filein, times):
    """
    save data
    :param times:times run the function
    :return:
    """
    df = fa_to_df(filein)
    seq, label = getseq(df)
    probMatr, tag = AAindexVector(seq, label)
    trainX = probMatr
    trainY = tag
    print('time %d has been saved' % times)
    fileX = open('./newdata/trainX%d.pickle' % times, 'wb')
    fileY = open('./newdata/trainY%d.pickle' % times, 'wb')
    pickle.dump(trainX, fileX, protocol=4)
    pickle.dump(trainY, fileY, protocol=4)
    fileX.close()
    fileY.close()

def load_data_train(times):
    """
    lode data
    :param times: times run the function
    :return:
    """
    with open('./newdata/trainX%d.pickle' % times, 'rb') as file1:
        trainX = pickle.load(file1)
    with open('./newdata/trainY%d.pickle' % times, 'rb') as file2:
        trainY = pickle.load(file2)
    input_row = trainX.shape[2]
    input_col = trainX.shape[3]
    trainX.shape = (trainX.shape[0], input_row, input_col)
    input = Input(shape=(input_row, input_col))
    trainY = kutils.to_categorical(trainY)
    return trainX, trainY, input

def save_data_test(filein, times):
    """
    save data
    :param filein:
    :return:
    """
    df = fa_to_df(filein)
    seq, label = getseq(df)
    probMatr, tag = AAindexVector(seq, label)
    testX = probMatr
    testY = tag
    fileX = open('./newdata/testX%d.pickle' % times, 'wb')
    fileY = open('./newdata/testY%d.pickle' % times, 'wb')
    pickle.dump(testX, fileX, protocol=4)
    pickle.dump(testY, fileY, protocol=4)
    fileX.close()
    fileY.close()

def load_data_test(times):
    """
    lode data
    :return:
    """
    with open('./newdata/testX%d.pickle' % times, 'rb') as file1:
        testX = pickle.load(file1)
    with open('./newdata/testY%d.pickle' % times, 'rb') as file2:
        testY = pickle.load(file2)
    input_row = testX.shape[2]
    input_col = testX.shape[3]
    testX.shape = (testX.shape[0], input_row, input_col)
    testY = kutils.to_categorical(testY)
    return testX, testY

def load_data(path):
    with open(path, 'rb') as file1:
        trainX = pickle.load(file1)
    with open(path.replace('X', 'Y'), 'rb') as file2:
        trainY = pickle.load(file2)
    input_row = trainX.shape[2]
    input_col = trainX.shape[3]
    trainX.shape = (trainX.shape[0], input_row, input_col)
    input = Input(shape=(input_row, input_col))
    trainY = kutils.to_categorical(trainY)
    return trainX, trainY, input

def load_data_two(path1, path2):
    trainX1, trainY1, input1 = load_data(path1)
    trainX2, trainY2, input2 = load_data(path2)
    trainX = np.vstack((trainX1, trainX2))
    trainY = np.vstack((trainY1, trainY2))
    index_list = [i for i in range(len(trainY))]
    random.shuffle(index_list)
    shuffled_seq = []
    shuffled_label = []
    for num in index_list:
        shuffled_seq.append(trainX[num])
        shuffled_label.append(trainY[num])
    shuffled_seq = np.array(shuffled_seq)
    shuffled_label = np.array(shuffled_label)
    return shuffled_seq, shuffled_label

def mygetseq_onehot(df):
    seq = []
    label = []
    for indexs in df.index:
        seq.append(df.loc[indexs].values[1])
        label.append(df.loc[indexs].values[0])
    tag = np.array(label)
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