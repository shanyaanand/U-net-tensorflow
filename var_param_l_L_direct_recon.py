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
import time
from sklearn.semi_supervised import label_propagation

color_val =spy_colors
color_val[0] = color_val[1]
color_val[1] = color_val[2]
color_val[2] = color_val[3]
color_val[3] = [0,0,0]

def loadfeat(curr_nmf):
    
    fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_subsampled_new/') 
    f = fullfile + curr_nmf
    feat = np.load(f)
    return feat['features'],feat['act_labels'],feat['pred_labels'],feat['test_pos'], feat['s0'],feat['s1'],feat['l'],feat['L_param']\
           ,feat['data_recon_err'],feat['label_recon_err']



start_time = time.time()
fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_subsampled_new/') 
rng = np.random.RandomState(2345)
bg_param =0.07
# rank =10
iter_val = 0

data = loaddata(1,bg_param)
s0 = data[2]
s1 = data[3]
tot_labels = data[1]
print tot_labels.shape
labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
shuff_labeled_pos = rng.permutation(labeled_pos)
size_labeled_set = len(labeled_pos)
train_pos = shuff_labeled_pos[:np.floor(0.9*size_labeled_set)]
test_pos = shuff_labeled_pos[np.floor(0.9*size_labeled_set):]

print np.histogram(tot_labels[labeled_pos],[0,1,2,3,4,5])
print np.histogram(tot_labels[train_pos],[0,1,2,3,4,5])
print np.histogram(tot_labels[test_pos],[0,1,2,3,4,5])

ssnmf_input_labels = np.empty_like(tot_labels)
np.copyto(ssnmf_input_labels, tot_labels)
ssnmf_input_labels[test_pos] = 0

iter_val = 14


for l in [0,0.5,3]:

    
    for L_param in [0.001]:
        if(os.path.exists(fullfile) == False):
            os.makedirs(fullfile)
        f = fullfile + ('direct_ssnmf_recon_%d.npz'%(iter_val))
        np.savez(f,features =feat_mat,act_labels=data[1],pred_labels = label_mat,test_pos = test_pos, s0 = data[2],s1 = data[3],
                 l=l,L_param = L_param,data_recon_err = data_recon_err,label_recon_err=label_recon_err)

        y_pred = np.argmax(label_mat,axis=0)
        labels = y_pred.reshape(s0,s1)
        f2 = fullfile + 'labels_direct_ssnmf_recon_%d.jpg'%(iter_val)
        save_rgb(f2, labels,colors=color_val )

        iter_val+=1
        print("--- %s minutes ---" % ((time.time() - start_time)/60.))

        start_time = time.time()


