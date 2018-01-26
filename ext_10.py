
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

#
#def loadfeat(curr_nmf):
#    
#    fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_tr14_te5/') 
#    f = fullfile +curr_nmf
#    feat = np.load(f)
#    return feat['features'],feat['label_mat'],feat['tot_labels'],feat['test_pos'],\
#           feat['data_recon_err'],feat['label_recon_err']
#
#start_time = time.time()
#rng = np.random.RandomState(2345)
fullfile = os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_tr14_te5/' 
#train_images = [1,2,3,4,5,6,7,12,13,14,15,16,17,19]       #   Skipping number 11 (Hyperplasia Case 1) due to lack of labels
#test_images = [8,9,10,18,20]      #        this is the test dataset which was commented as we did not have so much time . pLease remove both the # for actual work 
#rng = np.random.RandomState(2345)
#bg_param =0.07
#relax_label = True
#
#print "Processing training images"
#for noi,i in enumerate(train_images):
#    print "\tCurrently on",noi
#    data_pi = loaddata(i,0.07)
#    data_oi = data_pi[0]
#    tot_labels = data_pi[1]
#    print "\tTotal labels shape",tot_labels.shape
#    # print "\tTotal labels hist",np.histogram(tot_labels,range(6))
#    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
#    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
#    bg_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0 or tot_labels[x] == 4]
#    non_bg_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0 and tot_labels[x]!=4]
#    rng.shuffle(bg_pos)
#    print "Num labelled",len(non_bg_pos)
#    print "Num bg chosen",min(80000, len(bg_pos))
#    labeled_pos_tr = non_bg_pos + bg_pos[:min(80000, len(bg_pos))]
#    
#    shuff_labeled_pos = rng.permutation(labeled_pos_tr)
#    if noi == 0:
#        data_train = data_oi[:,shuff_labeled_pos]
#        label_train = tot_labels[shuff_labeled_pos]
#    else:
#        data_train = np.hstack((data_train,data_oi[:,shuff_labeled_pos]))
#        label_train = np.hstack((label_train,tot_labels[shuff_labeled_pos]))
#
#print "Processing testing images"        
#for noi,i in enumerate(test_images):
#    print "Currently on",noi
#    data_pi = loaddata(i,0.07)
#    data_oi = data_pi[0]
#    tot_labels = data_pi[1]
#    print "\tTotal labels shape",tot_labels.shape
#    # print "\tTotal labels hist",np.histogram(tot_labels,range(6))
#    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
#    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
#    bg_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] == 4]
#    non_bg_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0 and tot_labels[x]!=4]
#    rng.shuffle(bg_pos)
#    labeled_pos_tr = non_bg_pos + bg_pos[:min(40000, len(bg_pos))]    
#    shuff_labeled_pos = rng.permutation(labeled_pos_tr)
#    print "Num labelled",len(non_bg_pos)
#    print "Num bg chosen",min(40000, len(bg_pos))
#    if noi == 0:
#        data_test = data_oi[:,shuff_labeled_pos]
#        label_test = tot_labels[shuff_labeled_pos]
#    else:
#        data_test = np.hstack((data_test,data_oi[:,shuff_labeled_pos]))
#        label_test = np.hstack((label_test,tot_labels[shuff_labeled_pos]))
#
#shufftr = range(data_train.shape[1])
#shuffte = range(data_test.shape[1])
#rng.shuffle(shufftr)
#rng.shuffle(shuffte)
#
#data_train = data_train[:,shufftr]
#label_train = label_train[shufftr]
#data_test = data_test[:,shuffte]
#label_test = label_test[shuffte]
#
#ssnmf_input_data = np.hstack((data_train,data_test))
#ssnmf_input_label = np.hstack((label_train,np.zeros_like(label_test)))
#tot_labels =  np.hstack((label_train,label_test))
#
#test_pos = [x for x in range(ssnmf_input_label.shape[0]) if (ssnmf_input_label[x] == 0  and tot_labels[x] !=0)]
#
lambda_vals = [0.5]
lparam_vals = [0.01]
rank_list = [4, 7, 10, 13, 16]
lij_lambda_vals = [] #[(0, 0), (0.0001, 0.5), (0.005, 0.5), (0.001, 0.5)] #(0.005, 1), 
for x in lambda_vals:
    for y in lparam_vals:
        lij_lambda_vals.append( (y, x) )

lij_lambda_vals = lij_lambda_vals

res_file = fullfile + "svm_values.csv"

for rank in rank_list:
	for l_pair in lij_lambda_vals:
		data = np.load(fullfile + "label_feat_dump_rank_" + str(rank) + "_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz")
	
	        feat_mat,label_mat, tot_labels, test_pos = data['features'], data['label_mat'], data['tot_labels'], data['test_pos']
	        act_labels = tot_labels[test_pos]
		train_pos = np.delete( np.array(range(feat_mat.shape[1])), test_pos )

        	print rank, l_pair[0],l_pair[1]
	
	        lp_in_labels =np.empty_like(tot_labels)
	        np.copyto(lp_in_labels,tot_labels)
	        lp_in_labels[test_pos] = 0
	        lp_in_labels -= 1

#        clf = label_propagation.LabelSpreading(kernel='knn', alpha=1)
#        clf.fit(np.transpose(feat_mat), lp_in_labels)
#        y_pred_lp = clf.transduction_    
#        pred_labels_lp = y_pred_lp + 1
	        clf_svm = svm.SVC()
	        clf_svm.fit(np.transpose(feat_mat[:,train_pos]), tot_labels[train_pos])
	        tr_pred = clf_svm.predict(np.transpose(feat_mat[:,train_pos]))
	        y_pred_lp = clf_svm.predict(np.transpose(feat_mat[:,test_pos]))
	        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
	        tr_fs = metrics.f1_score(tot_labels[train_pos], tr_pred, average='macro')

        	print 'testing acc'
	        print f_s
	        print 'train acc '
	        print tr_fs
        
        	with open(res_file, "a+") as res:
	            res.write(str(rank) + "," + str(l_pair[0]) + "," + str(l_pair[1]) + "," + str(tr_fs) + "," + str(f_s) + "\n")
