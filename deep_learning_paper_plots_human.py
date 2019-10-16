import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import data_processing as dp
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
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
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
import seaborn as sns
import fine_tuning_mouse as ft
import re

from Bio import SeqIO
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (25.0, 20.0)
plt.rcParams.update({'font.size': 42})

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)

def process_pseui(fileloc):
    PSEUI_res_dict = dict()
    fin = open(fileloc,'r')
    for line in fin:
        string_s = line.strip()
        if string_s.startswith('The') or string_s.startswith('Query_seqnum'):
            continue
        mylist = string_s.split(' ')
        psu_site = mylist[-1]
        seqnum = int(mylist[0])
        PSEUI_res_dict.setdefault(seqnum,[]).append(psu_site)
    return PSEUI_res_dict

def read_sequence(fastafile):
    True_dict = dict()
    counter = 0
    for index, record in enumerate(SeqIO.parse(fastafile, 'fasta')):
        counter+=1
        seqid = str(record.id)
        seqlist = seqid.split('_')
        seqclass = seqlist[-1]
        True_dict[counter] = int(seqclass)
    return True_dict,counter

def get_true_classes(true_dict,counter):
    true_class_list = []
    for i in range(1,counter):
        true_class_list.append(true_dict[i])
    return true_class_list

def get_predicted_classes(pred_dict,counter):
    pseui_predicted_list = []
    for i in range(1,counter):
        if i in pred_dict.keys():
            if(len(pred_dict[i])>0):
                if '11' in pred_dict[i]:
                    pseui_predicted_list.append(1)
                else:
                    pseui_predicted_list.append(0)
        else:
            pseui_predicted_list.append(0)
    return pseui_predicted_list

def get_predicted_classes_irna(pred_dict,counter):
    pseui_predicted_list = []
    for i in range(1,counter):
        pseui_predicted_list.append(pred_dict[i])
    return pseui_predicted_list

def process_irnapsu(fileloc):
    irnapsu_res_dict = dict()
    counter = 0
    seqnum2 = ''
    fin = open(fileloc,'r')
    for line in fin:
        string_s = line.strip()
##        print(string_s)
        counter+=1
        if((counter%2)!=0):
            seqlist = re.split('\s+',string_s)
            seqnum = seqlist[1]
            seqnum2 = seqnum.replace('#','')
        elif(string_s.startswith('None')):
             irnapsu_res_dict[int(seqnum2)] = 0
        else:
             strlist =  re.split('\s+',string_s)
             U_pos = strlist[5]
             if(int(U_pos)==11):
                 irnapsu_res_dict[int(seqnum2)] = 1
             else:
                 irnapsu_res_dict[int(seqnum2)] = 0            
    return irnapsu_res_dict
##0-secstr_seq_denseconcat_60perc_best.h5
def my_roc_curve(prefix,datatype):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    sns.set(font_scale = 1.2)
    colors = sns.hls_palette(10)
    fig, ax = plt.subplots()
    fig.set_size_inches(12.0, 8.0)
    for i in range(10):
        testfile = prefix+str(i)+".txt"
        test = ft.read_composite_data(testfile)
        test_shuffled = test.sample(frac=1)
        seq_test,secstr_test,label_test = ft.get_all_seq(test)
        testX2,testY = ft.onehotkey(seq_test,label_test)
        testX1,testY = ft.onehotkey_sec(secstr_test,label_test)
        testY = np_utils.to_categorical(testY,2)
        testY = testY.reshape(-1,2)
        row1,col1 = testX1[0].shape
        row2,col2 = testX2[0].shape
        testX1.shape = (testX1.shape[0],row1,col1)
        testX2.shape = (testX2.shape[0],row2,col2)
        cnn = models.load_model('%d-secstr_seq_denseconcat_60perc_best.h5' % i) #0-yeast_pretrain-merge.h5
        color = colors[i]
        probas_ = cnn.predict([testX1,testX2])
        fpr,tpr,thresholds = roc_curve(testY[:,1],probas_[:,1])
        tprs.append(np.interp(mean_fpr,fpr,tpr))
        roc_auc = auc(fpr,tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',label='Base line')
    mean_tpr = np.mean(tprs,axis=0)
    mean_auc = auc(mean_fpr,mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(datatype) +' Human data 10fold ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(str(datatype) + '_Human_10fold_ROC_curve.tiff')
    

def prec_rec_curve(prefix,datatype,pred_thr):
    mydict_metrics = defaultdict(list)
    y_real = []
    y_proba = []
    sns.set(font_scale = 1.2)
    colors = sns.hls_palette(10)
    fig, ax = plt.subplots()
    fig.set_size_inches(12.0, 8.0)
    lw = 2
    i = 0
    precision_bal = []
    recall_bal = []
    threshold_bal = []
    f1_bal = []
    mcc_bal = []
    sens_bal = []
    spec_bal = []
    acc_bal = []
    auc_bal = []
    print(prefix)
    for j in range(5):
        for i in range(10):
            testfile = prefix+str(i)+".txt"
            test = ft.read_composite_data(testfile)
            test_shuffled = test.sample(frac=1)
            seq_test,secstr_test,label_test = ft.get_all_seq(test)
            testX2,testY = ft.onehotkey(seq_test,label_test)
            testX1,testY = ft.onehotkey_sec(secstr_test,label_test)
            testY = np_utils.to_categorical(testY,2)
            testY = testY.reshape(-1,2)
            row1,col1 = testX1[0].shape
            row2,col2 = testX2[0].shape
            testX1.shape = (testX1.shape[0],row1,col1)
            testX2.shape = (testX2.shape[0],row2,col2)
            cnn = models.load_model('%d-secstr_seq_denseconcat_60perc_best.h5' % i)
            color = colors[i]
            probas_ = cnn.predict([testX1,testX2])
            pre_score = cnn.evaluate([testX1,testX2], testY, batch_size=32, verbose=0)
            acc_bal.append(pre_score[1])
            pred_score = probas_[:,1]
            true_class = testY[:,1]
            precision, recall, thresholds = precision_recall_curve(true_class, pred_score)
            # plt.plot(recall, precision, lw=lw, color=color,label='Fold %d (area = %0.3f)' % (i, auc(recall, precision)))
            y_real.append(true_class)
            y_proba.append(pred_score)
            threshold_bal.append(np.median(thresholds, axis=0))
            scores = np.where(pred_score > pred_thr, 1, 0)
            tn, fp, fn, tp = confusion_matrix(true_class, scores).ravel()
            specificity =  tn / (tn+fp)
            sensitivity = tp/(tp+fn)
            precision_s = precision_score(true_class, scores)
            recall_s = recall_score(true_class, scores)
            precision_bal.append(precision_s)
            recall_bal.append(recall_s)
            f1_s = f1_score(true_class, scores)
            mcc = matthews_corrcoef(true_class,scores)
            f1_bal.append(f1_s)
            mcc_bal.append(mcc)
            sens_bal.append(sensitivity)
            spec_bal.append(specificity)
    ##        threshold_bal.append(thresholds)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)    
    precision, recall, thresholds = precision_recall_curve(y_real, y_proba)
    # plt.plot(recall, precision, lw=lw, color='g', linestyle='--',label='Mean (area = %0.3f)' % (auc(recall, precision)))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(str(datatype) +' Human data 10fold Precision Recall Curve')
    # plt.legend(loc="lower left")
    # plt.savefig(str(datatype) + '_Human_10fold_Precision-Recall_curve.tiff')
    mydict_metrics['precision'] = precision_bal
    mydict_metrics['recall'] = recall_bal
    mydict_metrics['f1'] = f1_bal
    mydict_metrics['mcc'] = mcc_bal
    mydict_metrics['sensitivity'] = sens_bal
    mydict_metrics['specificity'] = spec_bal
    mydict_metrics['accuracy'] = acc_bal
    mydict_metrics['auc'] = auc_bal
    mydict_metrics['threshold'] = threshold_bal
    return mydict_metrics

def pseui_result_metrics(result_prefix,fasta_prefix,datatype):
    mydict_metrics = defaultdict(list)
    precision_bal = []
    recall_bal = []
    threshold_bal = []
    f1_bal = []
    mcc_bal = []
    sens_bal = []
    spec_bal = []
    acc_bal = []
    auc_bal = []
    for i in range(10):
        result_file = result_prefix+str(i)+".txt"
        fasta_file = fasta_prefix+str(i)+"_clipped.fasta" #0_clipped.fasta
        result_dict = process_pseui(result_file)
        true_dict,count = read_sequence(fasta_file)
        true_class_lst = get_true_classes(true_dict,count)
        pred_cls_lst = get_predicted_classes(result_dict,count)
        pseui_predicted_nparr = np.array(pred_cls_lst)
        true_class_nparr = np.array(true_class_lst)
        mcc_bal.append(matthews_corrcoef(true_class_nparr,pseui_predicted_nparr))
        f1_bal.append(f1_score(true_class_nparr,pseui_predicted_nparr))
        precision_bal.append(precision_score(true_class_nparr,pseui_predicted_nparr))
        recall_bal.append(recall_score(true_class_nparr,pseui_predicted_nparr))
        fpr, tpr, thresholds = roc_curve(true_class_nparr,pseui_predicted_nparr)
        auc_bal.append(auc(fpr, tpr))
        acc_bal.append(accuracy_score(true_class_nparr,pseui_predicted_nparr))
        tn, fp, fn, tp = confusion_matrix(true_class_nparr, pseui_predicted_nparr).ravel()
        spec_bal.append(tn / (tn+fp))
        sens_bal.append(tp/(tp+fn))
    mydict_metrics['precision'] = precision_bal
    mydict_metrics['recall'] = recall_bal
    mydict_metrics['f1'] = f1_bal
    mydict_metrics['mcc'] = mcc_bal
    mydict_metrics['sensitivity'] = sens_bal
    mydict_metrics['specificity'] = spec_bal
    mydict_metrics['accuracy'] = acc_bal
    mydict_metrics['auc'] = auc_bal
    return mydict_metrics
##human_irna_pseu_results_fold_0.txt
##human_PESUI_results_fold_0.txt
def irnapsu_result_metrics(result_prefix,fasta_prefix,datatype):
    mydict_metrics = defaultdict(list)
    mcc_list = []
    f1_list = []
    prec_lst = []
    recall_lst = []
    auc_lst = []
    acc_lst = []
    sens_lst = []
    specs_lst = []
    for i in range(10):
        fasta_file = fasta_prefix+str(i)+'_clipped.fasta' #0_clipped.fasta
        result_file = result_prefix+str(i)+".txt"
        result_dict = process_irnapsu(result_file)
        true_dict,count = read_sequence(fasta_file)
        true_class_lst = get_true_classes(true_dict,count)
        pred_cls_lst = get_predicted_classes_irna(result_dict,count)
        pseui_predicted_nparr = np.array(pred_cls_lst)
        true_class_nparr = np.array(true_class_lst)
        mcc_list.append(matthews_corrcoef(true_class_nparr,pseui_predicted_nparr))
        f1_list.append(f1_score(true_class_nparr,pseui_predicted_nparr))
        prec_lst.append(precision_score(true_class_nparr,pseui_predicted_nparr))
        recall_lst.append(recall_score(true_class_nparr,pseui_predicted_nparr))
        fpr, tpr, thresholds = roc_curve(true_class_nparr,pseui_predicted_nparr)
        auc_lst.append(auc(fpr, tpr))
        acc_lst.append(accuracy_score(true_class_nparr,pseui_predicted_nparr))
        tn, fp, fn, tp = confusion_matrix(true_class_nparr, pseui_predicted_nparr).ravel()
        specs_lst.append(tn / (tn+fp))
        sens_lst.append(tp/(tp+fn))
##    print("precision ",prec_lst)
##    print("recall ",recall_lst)
##    print("F1 score ",f1_list)
##    print("mcc score ",mcc_list)
##    print("sensitivity ",sens_lst)
##    print("specificity ",specs_lst)
##    print("accuracy ",acc_lst)
##    print("auc score ",auc_lst)
    mydict_metrics['precision'] = prec_lst
    mydict_metrics['recall'] = recall_lst
    mydict_metrics['f1'] = f1_list
    mydict_metrics['mcc'] = mcc_list
    mydict_metrics['sensitivity'] = sens_lst
    mydict_metrics['specificity'] = specs_lst
    mydict_metrics['accuracy'] = acc_lst
    mydict_metrics['auc'] = auc_lst
##    print(mydict_metrics)
    return mydict_metrics

def violin_plots(deep_pseu,irna_psu,pseui,outprefix,datatype):
    sns.set(font_scale = 2)
    colors = sns.hls_palette(10)
    fig, ax = plt.subplots()
    score_ls = []
    mean_metrics_file = outprefix+"mean.csv"
    violin_plot_file = outprefix+"violin_plot.tiff"
    for i in deep_pseu.keys():
        for j in deep_pseu[i]:
            score_ls.append(['Deep_learning_model', i,j])
    for i in pseui.keys():
        for j in pseui[i]:
            score_ls.append(['PseUI', i,j])
    for i in irna_psu.keys():
        for j in irna_psu[i]:
            score_ls.append(['iRNA-PseU', i,j])
    metrics_df = pd.DataFrame(score_ls,columns=['Classifier','Score_type','Score'])
    mean_score_df = metrics_df.groupby(['Classifier','Score_type'],as_index=False)["Score"].mean()
    mean_score_df.to_csv(mean_metrics_file, encoding='utf-8', index=False) 
    selected_metrics = ['precision','recall','f1','mcc','accuracy']
    subset_df = metrics_df.loc[metrics_df['Score_type'].isin(selected_metrics)]
    ax = sns.violinplot(x="Classifier", y="Score", hue="Score_type", data=subset_df, linewidth=6) #producing the boxplot
    plt.setp(ax.get_legend().get_texts(), fontsize='30') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='35') # for legend title
    ax.set_title("Comparison of Pseudouridine site prediction methods on "+datatype)
    ax.figure.savefig(violin_plot_file)
    
##dict_keys(['precision', 'recall', 'f1', 'mcc', 'sensitivity', 'specificity', 'accuracy'])
##./webserver_data_mouse/balanced_data/balanced_mouse_emboss_60perc_testing_0_clipped.fasta
##0.txt
metrics_dict_balanced = prec_rec_curve("./webserver_test_data_balanced/balanced_emboss_60perc_testing_","balanced",0.35)
metrics_dict_imbalanced = prec_rec_curve("./webserver_test_data_imbalanced/emboss_60perc_testing_","imbalanced",0.35)
##emboss_60perc_testing_0.txt
##0.txt
# my_roc_curve("./webserver_test_data_balanced/balanced_emboss_60perc_testing_","balanced")
# my_roc_curve("./webserver_test_data_imbalanced/emboss_60perc_testing_","imbalanced")
#webserver_test_data_balanced/human_PESUI_results_fold_0.txt
#webserver_test_data_balanced/human_irna_pseu_results_fold_
Pseui_metrics_balanced = pseui_result_metrics("./webserver_test_data_balanced/human_PESUI_results_fold_","./webserver_test_data_balanced/balanced_emboss_60perc_testing_","balanced")
irna_psu_metrics_bal = irnapsu_result_metrics("./webserver_test_data_balanced/human_irna_pseu_results_fold_","./webserver_test_data_balanced/balanced_emboss_60perc_testing_","balanced")
violin_plots(metrics_dict_balanced,irna_psu_metrics_bal,Pseui_metrics_balanced,"balanced_dataset_comparison_","balanced data")

Pseui_metrics_imbalanced = pseui_result_metrics("./webserver_test_data_imbalanced/PSEU_results_fold_","./webserver_test_data_imbalanced/emboss_60perc_testing_","imbalanced") 
irna_psu_metrics_imbal = irnapsu_result_metrics("./webserver_test_data_imbalanced/irna_pseu_fold_","./webserver_test_data_imbalanced/emboss_60perc_testing_","imbalanced")
violin_plots(metrics_dict_imbalanced,irna_psu_metrics_imbal,Pseui_metrics_imbalanced,"imbalanced_dataset_comparison_","imbalanced data")


pickle.dump( metrics_dict_balanced, open( "deep_learning_balanced_results.pickle", "wb" ) )
pickle.dump( metrics_dict_imbalanced, open( "deep_learning_imbalanced_results.pickle", "wb" ) )

pickle.dump( Pseui_metrics_balanced, open( "Pseui_metrics_balanced_results.pickle", "wb" ) )
pickle.dump( Pseui_metrics_imbalanced, open( "Pseui_metrics_imbalanced_results.pickle", "wb" ) )

pickle.dump( irna_psu_metrics_bal, open( "irna_psu_metrics_balanced_results.pickle", "wb" ) )
pickle.dump( irna_psu_metrics_imbal, open( "irna_psu_metrics_imbalanced_results.pickle", "wb" ) )

##print("##########################Deep learning balanced#############################################")
##print(metrics_dict_balanced)
##print("############################PSEUI balanced###########################################")
##print(Pseui_metrics_balanced)
##print("############################irna pseu balanced###########################################")
##print(irna_psu_metrics_bal)

##
##with open('deep_imbal.pickle', 'wb') as handle:
##    pickle.dump(metrics_dict_imbalanced, handle, protocol=pickle.HIGHEST_PROTOCOL)
##
##with open('Pseui_metrics_imbal.pickle', 'wb') as handle:
##    pickle.dump(Pseui_metrics_imbalanced, handle, protocol=pickle.HIGHEST_PROTOCOL)
##
##with open('irna_psu_imbal.pickle', 'wb') as handle:
##    pickle.dump(irna_psu_metrics_imbal, handle, protocol=pickle.HIGHEST_PROTOCOL)    
