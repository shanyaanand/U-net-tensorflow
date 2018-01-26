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
train_images = [1,2,3,4,5,6,7,12,13,14,15,16,17,19]       #   Skipping number 11 (Hyperplasia Case 1) due to lack of labels
test_images = [8,9,10,18,20]      #        this is the test dataset which was commented as we did not have so much time . pLease remove both the # for actual work 
rng = np.random.RandomState(2345)
bg_param =0.07
relax_label = True

print "Processing training images"
for noi,i in enumerate(train_images):
    print "\tCurrently on",noi
    data_pi = loaddata(i,0.07)
    data_oi = data_pi[0]
    tot_labels = data_pi[1]
    print "\tTotal labels shape",tot_labels.shape
    # print "\tTotal labels hist",np.histogram(tot_labels,range(6))
    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
    bg_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0 or tot_labels[x] == 4]
    non_bg_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0 and tot_labels[x]!=4]
    rng.shuffle(bg_pos)
    print "Num labelled",len(non_bg_pos)
    print "Num bg chosen",min(80000, len(bg_pos))
    labeled_pos_tr = non_bg_pos + bg_pos[:min(80000, len(bg_pos))]
    
    shuff_labeled_pos = rng.permutation(labeled_pos_tr)
    if noi == 0:
        data_train = data_oi[:,shuff_labeled_pos]
        label_train = tot_labels[shuff_labeled_pos]
    else:
        data_train = np.hstack((data_train,data_oi[:,shuff_labeled_pos]))
        label_train = np.hstack((label_train,tot_labels[shuff_labeled_pos]))

print "Processing testing images"        
for noi,i in enumerate(test_images):
    print "Currently on",noi
    data_pi = loaddata(i,0.07)
    data_oi = data_pi[0]
    tot_labels = data_pi[1]
    print "\tTotal labels shape",tot_labels.shape
    # print "\tTotal labels hist",np.histogram(tot_labels,range(6))
    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
    bg_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] == 4]
    non_bg_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0 and tot_labels[x]!=4]
    rng.shuffle(bg_pos)
    labeled_pos_tr = non_bg_pos + bg_pos[:min(40000, len(bg_pos))]    
    shuff_labeled_pos = rng.permutation(labeled_pos_tr)
    print "Num labelled",len(non_bg_pos)
    print "Num bg chosen",min(40000, len(bg_pos))
    if noi == 0:
        data_test = data_oi[:,shuff_labeled_pos]
        label_test = tot_labels[shuff_labeled_pos]
    else:
        data_test = np.hstack((data_test,data_oi[:,shuff_labeled_pos]))
        label_test = np.hstack((label_test,tot_labels[shuff_labeled_pos]))

shufftr = range(data_train.shape[1])
shuffte = range(data_test.shape[1])
rng.shuffle(shufftr)
rng.shuffle(shuffte)

data_train = data_train[:,shufftr]
label_train = label_train[shufftr]
data_test = data_test[:,shuffte]
label_test = label_test[shuffte]

ssnmf_input_data = np.hstack((data_train,data_test))
ssnmf_input_label = np.hstack((label_train,np.zeros_like(label_test)))
tot_labels =  np.hstack((label_train,label_test))

test_pos = [x for x in range(ssnmf_input_label.shape[0]) if (ssnmf_input_label[x] == 0  and tot_labels[x] !=0)]

lambda_vals = [0.1, 0.5, 1, 2, 10]
lparam_vals = [0.0001, 0.001, 0.005, 0.01, 0.1]

lij_lambda_vals = [] #[(0, 0), (0.0001, 0.5), (0.005, 0.5), (0.001, 0.5)] #(0.005, 1), 
for x in lambda_vals:
    for y in lparam_vals:
        lij_lambda_vals.append( (y, x) )

lij_lambda_vals = [(0,0)] + lij_lambda_vals

rank_vals = [4,7,10,13,16]
rank_skip_list = [10]
res_file = fullfile + "rank_results.csv"

with open(res_file, "a+") as res:
    res.write("rank,L_ij,lambda,testing_fs\n")

for l_pair in lij_lambda_vals:
    for rank in rank_vals:
        print "Processing",rank,l_pair[0],l_pair[1]

        if rank in rank_skip_list:           
            if os.path.isfile(fullfile + "label_feat_dump_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz") == True:
                print "File already exists...skipping"
                os.rename(fullfile + "label_feat_dump_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz", fullfile + "label_feat_dump_rank_10_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz")
                continue

        
        fname = fullfile + "label_feat_dump_rank_" + str(rank) + "_" + str(l_pair[0]) + "_" + str(l_pair[1]) + ".npz"
        if os.path.isfile(fname) == True:
            print "File already exists...skipping"
            continue
        
        feat_mat,label_mat,data_recon_err,label_recon_err,eval_s = ssnmf(ssnmf_input_data,ssnmf_input_label,rank,l_pair[1],relax_label = True,L_param=l_pair[0])
        act_labels = tot_labels[test_pos]

        print l_pair[0],l_pair[1],data_recon_err,label_recon_err

        lp_in_labels =np.empty_like(tot_labels)
        np.copyto(lp_in_labels,tot_labels)
        lp_in_labels[test_pos] = 0
        lp_in_labels -= 1

        clf = label_propagation.LabelSpreading(kernel='knn', alpha=1)
        clf.fit(np.transpose(feat_mat), lp_in_labels)
        y_pred_lp = clf.transduction_    
        pred_labels_lp = y_pred_lp + 1
        pred_labels = pred_labels_lp[test_pos]

        np.savez(fname,features=feat_mat,label_mat = label_mat,tot_labels=tot_labels, input_labels = ssnmf_input_label ,
                 test_pos = test_pos,data_recon_err = data_recon_err,label_recon_err=label_recon_err)    

        f_s =  metrics.f1_score(act_labels, pred_labels, average='macro')
        # tr_fsc = metrics.f1_score(tot_labels[train_pos], pred_labels_lp[train_pos], average='macro')
        with open(res_file, "a+") as res:
            res.write(str(rank) + "," + str(l_pair[0]) + "," + str(l_pair[1]) + "," + str(f_s) + "\n")

        print 'testing acc'
        print f_s,'\n'
        print("--- %s minutes ---" % ((time.time() - start_time)/60.))
print("--- %s minutes ---" % ((time.time() - start_time)/60.))


