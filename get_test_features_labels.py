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

start_time = time.time()
rng = np.random.RandomState(2345)
fullfile = os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_subsampled_new/' 

lij_lambda_vals = [(0.005, 1), (0.0001, 1), (0, 0), (0.001, 0.5),] 
rank_vals = [4,7,10,13,16]
rank_skip_list = [10]
res_file = fullfile + "rank_results.csv"

for l_pair in lij_lambda_vals:
    for rank in rank_vals:
        if rank in rank_skip_list:
            continue   
        l_file = np.load(fullfile + "label_feat_dump_rank_" + str(rank) + "_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz")

        feat_mat,tot_labels, test_pos = l_file['features'], l_file['tot_labels'], l_file['test_pos']
        act_labels = tot_labels[test_pos]
        print np.shape(feat_mat[:, test_pos])
        print np.shape(act_labels)
        lp_in_labels =np.empty_like(tot_labels)
        np.copyto(lp_in_labels,tot_labels)
        lp_in_labels[test_pos] = 0
        lp_in_labels -= 1
        print "Performing label spreading on",str(rank),str(l_pair)
        clf = label_propagation.LabelSpreading(kernel='knn', alpha=1)
        clf.fit(np.transpose(feat_mat), lp_in_labels)
        y_pred_lp = clf.transduction_    
        pred_labels_lp = y_pred_lp + 1
        pred_labels = pred_labels_lp[test_pos]
        op_file = fullfile + "label_feat_test_dump_" + str(rank) + "_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz"
        np.savez(op_file, test_feat=feat_mat[:, test_pos], act_labels=act_labels, pred_labels=pred_labels)
        print("--- %s minutes ---" % ((time.time() - start_time)/60.))
print("--- %s minutes ---" % ((time.time() - start_time)/60.))  