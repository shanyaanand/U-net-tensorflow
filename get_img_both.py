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
from skimage.color import label2rgb
colors = [(1, 0, 0), (0, 1, 0), (1, 1, 0)]
color_val =spy_colors
color_val[0] = color_val[1]
color_val[1] = color_val[2]
color_val[2] = color_val[3]
color_val[3] = [0,0,0]

def loadfeat(curr_nmf):    
    fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/direct_ssnmf_recon/') 
    f = fullfile +curr_nmf

    feat = np.load(f)
    return feat['features'],feat['act_labels'],feat['pred_labels'],feat['test_pos'], feat['s0'],feat['s1'],feat['l'],feat['L_param']\
           ,feat['data_recon_err'],feat['label_recon_err']

start_time = time.time()
rng = np.random.RandomState(2345)
bg_param =0.07
rank =10
iter_val = 0

for i in [0, 4]:
    for j in range(2):
        data = loaddata( (i+1 if j == 0 else i+11) ,bg_param)
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

        ssnmf_input_labels = np.empty_like(tot_labels)
        np.copyto(ssnmf_input_labels, tot_labels)
        ssnmf_input_labels[test_pos] = 0

        iter_val = 14
        for l, L_param in [(0.5, 0.01)]:
            feat_mat,label_mat,data_recon_err,label_recon_err,eval_s = ssnmf(data[0],ssnmf_input_labels,rank,l,relax_label = True,L_param=L_param)
            fullfile = os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_tr14_te5/' 
             
            if(os.path.exists(fullfile) == False):
                os.makedirs(fullfile)
            f = fullfile + ('direct_ssnmf_recon_%d.npz'%(iter_val))
            np.savez(f,features =feat_mat,act_labels=data[1],pred_labels = label_mat,test_pos = test_pos, s0 = data[2],s1 = data[3],
                     l=l,L_param = L_param,data_recon_err = data_recon_err,label_recon_err=label_recon_err)

            y_pred = np.argmax(label_mat,axis=0)
            labels = y_pred.reshape(s0,s1)          

            lp_in_labels =np.empty_like(tot_labels)
            np.copyto(lp_in_labels,tot_labels)
            lp_in_labels[test_pos] = 0
            lp_in_labels -= 1

            clf = label_propagation.LabelSpreading(kernel='knn', alpha=0.8)
            clf.fit(np.transpose(feat_mat), lp_in_labels)
            y_pred_lp = clf.transduction_
            
            pred_labels_lp = y_pred_lp[test_pos] + 1
            y_pred_lp = y_pred_lp + 1
            y_pred_lp[tot_labels == 0] = 0
            y_pred_lp[tot_labels == 4] = 0
            labels_lp = y_pred_lp.reshape(s0,s1) 
            l_pair = [l,L_param]
            # labels_lp = labels_lp + 1;
            # labels_lp[labels_lp==4] = 0

            np.savez(fullfile + "tot_labels_lp_recon_" + str(rank) + "_" + str(L_param) + "_" + str(l) + ".npz", tot_pred_labels=labels_lp) 
            if j == 0:
                f1 = fullfile + str(i+1) + 'normal__' + str(rank) + '_' + str(l_pair[0]) + '_' + str(l_pair[1]) + '.bmp'
            else:
                f1 = fullfile + str(i+11) + 'hyp__' + str(rank) + '_' + str(l_pair[0]) + '_' + str(l_pair[1]) + '.bmp'
            
            imgt = label2rgb(labels_lp,bg_label=0,bg_color=(0,0,0), colors=colors)
            newImg1 = pimg.fromarray(np.uint8(255*imgt))
            newImg1.save(f1,"BMP")    
               
            # f2 = fullfile + 'labels_direct_ssnmf_recon_normal%d.jpg'%(i)
            
            # labels = labels + 1
            # labels[labels==4] = 0 
            # imgt = label2rgb(labels,bg_label=0,bg_color=(0,0,0),colors = colors)
            # newImg1 = pimg.fromarray(np.uint8(255*imgt))
            # newImg1.save(f2,"BMP") 
            
            iter_val+=1
            print("--- %s minutes ---" % ((time.time() - start_time)/60.))
            start_time = time.time()