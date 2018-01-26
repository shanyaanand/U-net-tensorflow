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
    
    fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/nmf_feat_var_ranks/') 
    f = fullfile +curr_nmf

    feat = np.load(f)
    return feat['features'],feat['tot_labels'],feat['data_recon_err']

start_time = time.time()
rng = np.random.RandomState(2345)
fullfile = os.path.dirname(os.path.abspath(__file__))+ '/nmf_feat_var_ranks/' 

train_per = 90


for rank in [4,7,10,13,16,20]:

    feat_mat,tot_labels,data_recon_err=loadfeat('nmf_feat_rank_%d.npz'%(rank))


    print tot_labels.shape
    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
    shuff_labeled_pos = rng.permutation(labeled_pos)
    size_labeled_set = len(labeled_pos)

    train_pos = shuff_labeled_pos[:np.floor((train_per/100.)*size_labeled_set)]
    test_pos = shuff_labeled_pos[np.floor((train_per/100.)*size_labeled_set):]

    print np.histogram(tot_labels[labeled_pos],[0,1,2,3,4,5])
    print np.histogram(tot_labels[train_pos],[0,1,2,3,4,5])
    print np.histogram(tot_labels[test_pos],[0,1,2,3,4,5])

    
 

    act_labels = tot_labels[test_pos]



    print data_recon_err

    lp_in_labels =np.empty_like(tot_labels)
    np.copyto(lp_in_labels,tot_labels)
    lp_in_labels[test_pos] = 0
    lp_in_labels -= 1


    clf = label_propagation.LabelSpreading(kernel='knn', alpha=1)
    clf.fit(np.transpose(feat_mat), lp_in_labels)
    y_pred_lp = clf.transduction_
    
    pred_labels_lp = y_pred_lp + 1

    pred_labels = pred_labels_lp[test_pos]


    
    #print s0,s1,y_pred.shape


    print np.histogram(tot_labels[test_pos],[0,1,2,3,4,5])
    print np.histogram(pred_labels,[0,1,2,3,4,5])


    print np.histogram(tot_labels,[0,1,2,3,4,5])
    print np.histogram(pred_labels_lp,[0,1,2,3,4,5])
    f_s =  metrics.f1_score(act_labels, pred_labels, average='macro')
    c_mat =  metrics.confusion_matrix(act_labels, pred_labels)

    print 'testing acc'
    print f_s,'\n'
    print c_mat,'\n'

    print 'train acc '
    print metrics.f1_score(tot_labels[train_pos], pred_labels_lp[train_pos], average='macro')
    print metrics.confusion_matrix(tot_labels[train_pos], pred_labels_lp[train_pos])
   

    f = fullfile + ('res_ourmodel_%d.npz'%(train_per))
    np.savez(f,f_score =f_s,confusion_matrix = c_mat,data_recon_err=data_recon_err)

print("--- %s minutes ---" % ((time.time() - start_time)/60.))


