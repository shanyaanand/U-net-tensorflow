from loaddata import loaddata,loaddata_spatial, get_train_data
from ssnmf_func import ssnmf, nmf, ssfnnmf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image as pimg
import matplotlib
import os
from spectral import *
from sklearn import svm

from sklearn import metrics
import time
from sklearn.semi_supervised import label_propagation

color_val =spy_colors
color_val[0] = color_val[1]
color_val[1] = color_val[2]
color_val[2] = color_val[3]
color_val[3] = [0,0,0]

def loadfeat(curr_nmf):
    
    fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_subsampled_new/') 
    f = fullfile +curr_nmf
    feat = np.load(f)
    return feat['features'],feat['tot_labels'],feat['label_mat'],feat['test_pos'],\
           feat['data_recon_err'],feat['label_recon_err']

start_time = time.time()

data = loaddata(1,0.07)
s0 = data[2]
s1 = data[3]

fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_subsampled_new/') 
rng = np.random.RandomState(2345)

lij_lambda_vals = [(0.005, 1), (0.0001, 1), (0, 0), (0.001, 0.5),] 
rank_vals = [4,7,10,13,16]
rank_skip_list = [10]
lij_lambda_skip_list = [(2, 1), (1, 1), (0, 0)]

for l_pair in lij_lambda_vals:
    for rank in rank_vals:        
        if rank in rank_skip_list:
            continue
        print "Processing",rank,l_pair
        feat_mat,tot_labels,label_mat,test_pos,data_recon_err,label_recon_err=loadfeat("label_feat_dump_rank_" + str(rank) + "_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz")
        feat_mat, tot_labels, label_mat, test_pos = feat_mat[:, s0*s1], tot_labels[:, s0*s1], label_mat[:, s0*s1], test_pos[test_pos < s0*s1]
        act_labels = tot_labels[test_pos]
        print np.shape(tot_labels), s0, s1, (s0*s1)
        raw_input()
        y_pred = np.argmax(label_mat,axis =0)
        pred_labels = y_pred[test_pos] + 1
        # labels = y_pred.reshape(s0,s1)

        # target_names = ['class 0', 'class 1', 'class 2','back_gr']
        # print(metrics.classification_report(act_labels, pred_labels, target_names=target_names)),'\n'
        #print metrics.roc_auc_score(act_labels, pred_labels, average='macro'),'\n'
        # print metrics.f1_score(act_labels, pred_labels, average='macro'),'\n'
        # print metrics.confusion_matrix(act_labels, pred_labels),'\n'
        # print metrics.precision_recall_fscore_support(act_labels, pred_labels, average='macro'),'\n'
        # print metrics.normalized_mutual_info_score(act_labels, pred_labels),'\n'
        lp_in_labels =np.empty_like(tot_labels)
        np.copyto(lp_in_labels,tot_labels)
        lp_in_labels[test_pos] = 0
        lp_in_labels -= 1

        clf = label_propagation.LabelSpreading(kernel='knn', alpha=0.8)
        clf.fit(np.transpose(feat_mat), lp_in_labels)
        y_pred_lp = clf.transduction_
        
        pred_labels_lp = y_pred_lp[test_pos] + 1
        # print np.histogram(tot_labels[test_pos],[0,1,2,3,4,5])
        # print np.histogram(pred_labels,[0,1,2,3,4,5])
        # print np.histogram(pred_labels_lp,[0,1,2,3,4,5])

        # print(metrics.classification_report(act_labels, pred_labels_lp, target_names=target_names)),'\n'
        #print metrics.roc_auc_score(act_labels, pred_labels_lp, average='macro'),'\n'
        # print metrics.f1_score(act_labels, pred_labels_lp, average='macro'),'\n'
        # print metrics.confusion_matrix(act_labels, pred_labels_lp),'\n'
        # print metrics.precision_recall_fscore_support(act_labels, pred_labels_lp, average='macro'),'\n'
        # print metrics.normalized_mutual_info_score(act_labels, pred_labels_lp),'\n'
        
        np.savez(fullfile + "tot_labels_lp_recon_" + str(rank) + "_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz", tot_pred_labels=y_pred_lp)
        labels_lp = y_pred_lp.reshape(s0,s1)  
        f1 = fullfile + 'labels_lp_' + str(rank) + '_' + str(l_pair[0]) + '_' + str(l_pair[1]) + '.jpg'
        save_rgb(f1, labels_lp,colors=color_val )

        # f2 = fullfile + 'labels_direct_recon_' + str(rank) + '_' + str(l_pair[0]) + '_' + str(l_pair[1]) + '.jpg'
        # save_rgb(f2, labels,colors=color_val )
        print("--- %s minutes ---" % ((time.time() - start_time)/60.))

for rank in rank_skip_list:
    for i in range(len(lij_lambda_skip_list)):
        l_pair = lij_lambda_skip_list[i]
        print "Processing",rank,l_pair      
        feat_mat,tot_labels,label_mat,test_pos,data_recon_err,label_recon_err=loadfeat("label_feat_dump_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz")
        feat_mat, tot_labels, label_mat, test_pos = feat_mat[:, s0*s1], tot_labels[:, s0*s1], label_mat[:, s0*s1], test_pos[test_pos < s0*s1]
        act_labels = tot_labels[test_pos]
        y_pred = np.argmax(label_mat,axis =0)
        pred_labels = y_pred[test_pos] + 1
        # labels = y_pred.reshape(s0,s1)

        # target_names = ['class 0', 'class 1', 'class 2','back_gr']
        # print(metrics.classification_report(act_labels, pred_labels, target_names=target_names)),'\n'
        #print metrics.roc_auc_score(act_labels, pred_labels, average='macro'),'\n'
        # print metrics.f1_score(act_labels, pred_labels, average='macro'),'\n'
        # print metrics.confusion_matrix(act_labels, pred_labels),'\n'
        # print metrics.precision_recall_fscore_support(act_labels, pred_labels, average='macro'),'\n'
        # print metrics.normalized_mutual_info_score(act_labels, pred_labels),'\n'
        lp_in_labels =np.empty_like(tot_labels)
        np.copyto(lp_in_labels,tot_labels)
        lp_in_labels[test_pos] = 0
        lp_in_labels -= 1

        clf = label_propagation.LabelSpreading(kernel='knn', alpha=0.8)
        clf.fit(np.transpose(feat_mat), lp_in_labels)
        y_pred_lp = clf.transduction_
        
        pred_labels_lp = y_pred_lp[test_pos] + 1
        # print np.histogram(tot_labels[test_pos],[0,1,2,3,4,5])
        # print np.histogram(pred_labels,[0,1,2,3,4,5])
        # print np.histogram(pred_labels_lp,[0,1,2,3,4,5])

        # print(metrics.classification_report(act_labels, pred_labels_lp, target_names=target_names)),'\n'
        #print metrics.roc_auc_score(act_labels, pred_labels_lp, average='macro'),'\n'
        # print metrics.f1_score(act_labels, pred_labels_lp, average='macro'),'\n'
        # print metrics.confusion_matrix(act_labels, pred_labels_lp),'\n'
        # print metrics.precision_recall_fscore_support(act_labels, pred_labels_lp, average='macro'),'\n'
        # print metrics.normalized_mutual_info_score(act_labels, pred_labels_lp),'\n'
        
        np.savez(fullfile + "tot_labels_lp_recon_" + str(rank) + "_" + str(lij_lambda_vals[i][0]) + "_" + str(lij_lambda_vals[i][1]) + ".npz", tot_pred_labels=labels_lp)
        labels_lp = y_pred_lp.reshape(s0,s1)  
        f1 = fullfile + 'labels_lp_' + str(rank) + '_' + str(l_pair[0]) + '_' + str(l_pair[1]) + '.jpg'
        save_rgb(f1, labels_lp,colors=color_val )

        # f2 = fullfile + 'labels_direct_recon_' + str(rank) + '_' + str(l_pair[0]) + '_' + str(l_pair[1]) + '.jpg'
        # save_rgb(f2, labels,colors=color_val )
        print("--- %s minutes ---" % ((time.time() - start_time)/60.))



