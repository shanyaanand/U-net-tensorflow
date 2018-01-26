from loaddata import loaddata, get_train_data
from ssnmf_func import ssnmf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image as pimg
import matplotlib
import os
from sklearn.cluster import DBSCAN
from spectral import *
from sklearn import svm
import time
from sklearn.semi_supervised import label_propagation
from load_ssnmf_feat import ssnmf_feat

start_time = time.time()

color_val =spy_colors
color_val[0] = color_val[1]
color_val[1] = color_val[2]
color_val[2] = color_val[3]
color_val[3] = [0,0,0]
rng = np.random.RandomState(2345)
train_images = [1,2,3,4,5,8,9,10]
test_images = [6,7]
rank = 10
relax_label = True

for noi,i in enumerate(train_images):
    data_pi = loaddata(i,0.07)
    data_oi = data_pi[0]
    tot_labels = data_pi[1]
    print tot_labels.shape
    print np.histogram(tot_labels,range(6))
    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
    bg_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] == 4]
    non_bg_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0 and tot_labels[x]!=4]
    rng.shuffle(bg_pos)
    labeled_pos_tr = non_bg_pos + bg_pos[:15000]
    
    
    shuff_labeled_pos = rng.permutation(labeled_pos_tr)
    print np.histogram(tot_labels[shuff_labeled_pos],[0,1,2,3,4,5])
    if noi == 0:
        data_train = data_oi[:,shuff_labeled_pos]
        label_train = tot_labels[shuff_labeled_pos]
    else:
        data_train = np.hstack((data_train,data_oi[:,shuff_labeled_pos]))
        label_train = np.hstack((label_train,tot_labels[shuff_labeled_pos]))
        
for noi,i in enumerate(test_images):
    data_pi = loaddata(i,0.07)
    data_oi = data_pi[0]
    tot_labels = data_pi[1]
    print tot_labels.shape
    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
    shuff_labeled_pos = rng.permutation(labeled_pos)
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

fullfile = os.path.dirname(os.path.abspath(__file__))+ '/all_images_new_subsampled/' 
 
if(os.path.exists(fullfile) == False):
    os.makedirs(fullfile)

f = fullfile + ('data_tr8_te2_.npz')

np.savez(f,data_train=data_train,label_train=label_train,data_test=data_test,label_test=label_test)


print data_train.shape, label_train.shape, data_test.shape,label_test.shape
print np.histogram(label_train,range(6)),'\n'
print np.histogram(label_test,range(6))
