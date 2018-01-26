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
    
    fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_tr14_te5/') 
    f = fullfile +curr_nmf
    feat = np.load(f)
    return feat['features'],feat['label_mat'],feat['tot_labels'],feat['test_pos'],\
           feat['data_recon_err'],feat['label_recon_err']

start_time = time.time()
rng = np.random.RandomState(2345)
fullfile = os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_tr14_te5/' 

lambda_vals = [0.5, 1, 2, 0.1, 3, 10]         # To see the variations in regulartization parameter lambda
lparam_vals = [0.001, 0.0001, 0.005, 0.01, 0.1] # To see the variations in relaxation parameter L_ij

res_file = fullfile + "fs_results.csv"

for lnum,L_param in enumerate(lparam_vals):
    for inum,l in enumerate(lambda_vals):
        feat_mat,label_mat,tot_labels,test_pos,data_recon_err,label_recon_err=loadfeat('label_feat_dump_' + str(L_param) + '_' + str(l) + '.npz')
        act_labels = tot_labels[test_pos]
        print l,L_param,data_recon_err,label_recon_err        
        lp_in_labels =np.empty_like(tot_labels)
        np.copyto(lp_in_labels,tot_labels)
        lp_in_labels[test_pos] = 0
        lp_in_labels -= 1
        print "Fitting labels"
        clf = label_propagation.LabelSpreading(kernel='knn', alpha=1)
        clf.fit(np.transpose(feat_mat), lp_in_labels)
        print "Labels fit"
        y_pred_lp = clf.transduction_        
        print "Predicted values obtained"
        pred_labels_lp = y_pred_lp + 1
        pred_labels = pred_labels_lp[test_pos]        
        # print np.histogram(tot_labels[test_pos],[0,1,2,3,4,5])
        # print np.histogram(pred_labels,[0,1,2,3,4,5])
        # print np.histogram(tot_labels,[0,1,2,3,4,5])
        # print np.histogram(pred_labels_lp,[0,1,2,3,4,5])
        print "Calculating FScore"
        f_s =  metrics.f1_score(act_labels, pred_labels, average='macro')
        with open(res_file, "a+") as r:
            r.write(str(L_param) + "," + str(l) + "," + str(f_s) + "\n")
        print f_s
        # raw_input()