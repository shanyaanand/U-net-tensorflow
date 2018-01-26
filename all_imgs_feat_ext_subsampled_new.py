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
from sklearn.metrics import f1_score
#from load_ssnmf_feat import ssnmf_feat

start_time = time.time()

color_val = spy_colors
color_val[0] = color_val[1]
color_val[1] = color_val[2]
color_val[2] = color_val[3]
color_val[3] = [0,0,0]
rng = np.random.RandomState(2345)
# train_images = [1]
# test_images = [2]

#Parameters

train_images = [1,2,3,4,5,6,7,13,14,15,16,17,19]       #   Skipping number 11 (Hyperplasia Case 1) due to lack of labels
test_images = [8,9,10,18,20]      #        this is the test dataset which was commented as we did not have so much time . pLease remove both the # for actual work 
rng = np.random.RandomState(2345)
bg_param =0.07
rank = 10
relax_label = True
lambda_vals = [0.5, 1, 2, 0.1, 3, 10]         # To see the variations in regulartization parameter lambda
lparam_vals = [0.001, 0.0001, 0.005, 0.01, 0.1] # To see the variations in relaxation parameter L_ij
skip_list_lambda = []

# def loadalldata(curr_nmf):    
#     fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_subsampled/') 
#     f = fullfile +curr_nmf
#     feat = np.load(f)
#     return feat['data_train'],feat['label_train'],feat['data_test'],feat['label_test']

# def create_load_file():
print ("Processing training images")
for noi,i in enumerate(train_images):
    print ("\tCurrently on",noi)
    data_pi = loaddata(i,0.07)
    data_oi = data_pi[0]
    tot_labels = data_pi[1]
    print ("\tTotal labels shape",tot_labels.shape)
    print ("\tTotal labels hist",np.histogram(tot_labels,range(6)))
    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
    bg_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0 or tot_labels[x] == 4]
    non_bg_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0 and tot_labels[x]!=4]
    rng.shuffle(bg_pos)
    print ("Num labelled",len(non_bg_pos))
    print ("Num bg chosen",min(80000, len(bg_pos)))
    labeled_pos_tr = non_bg_pos + bg_pos[:min(80000, len(bg_pos))]
    
    shuff_labeled_pos = rng.permutation(labeled_pos_tr)
    print ("\tShuffled histogram",np.histogram(tot_labels[shuff_labeled_pos],[0,1,2,3,4,5]))
    if noi == 0:
        data_train = data_oi[:,shuff_labeled_pos]
        label_train = tot_labels[shuff_labeled_pos]
    else:
        data_train = numpy.hstack((data_train,data_oi[:,shuff_labeled_pos]))
        label_train = numpy.hstack((label_train,tot_labels[shuff_labeled_pos]))

print ("Processing testing images")        
for noi,i in enumerate(test_images):
    print ("Currently on",noi)
    data_pi = loaddata(i,0.07)
    data_oi = data_pi[0]
    tot_labels = data_pi[1]
    print ("\tTotal labels shape",tot_labels.shape)
    print ("\tTotal labels hist",np.histogram(tot_labels,range(6)))
    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
    bg_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] == 4]
    non_bg_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0 and tot_labels[x]!=4]
    rng.shuffle(bg_pos)
    labeled_pos_tr = non_bg_pos + bg_pos[:min(40000, len(bg_pos))]
    
    shuff_labeled_pos = rng.permutation(labeled_pos_tr)
    print ("Num labelled",len(non_bg_pos))
    print ("Num bg chosen",min(40000, len(bg_pos)))
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

fullfile = os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_tr14_te5/' 

if(os.path.exists(fullfile) == False):
    os.makedirs(fullfile)

f = fullfile + ('data_tr14_te5_1.npz')
np.savez(f,data_train=data_train,label_train=label_train,data_test=data_test,label_test=label_test)


# create_load_file()
# data_train,label_train,data_test,label_test = loadalldata('data_tr10_te10_.npz')

print ("train/label/test/label shapes:",data_train.shape, label_train.shape, data_test.shape,label_test.shape)
print ("train label histogram",np.histogram(label_train,range(6)),'\n')
print ("test label histogram",np.histogram(label_test,range(6)))

ssnmf_input_data = np.hstack((data_train,data_test))
print (ssnmf_input_data.shape)
ssnmf_input_label = np.hstack((label_train,np.zeros_like(label_test)))
print (ssnmf_input_label.shape)
tot_labels =  np.hstack((label_train,label_test))
print (tot_labels.shape)

test_pos = [x for x in range(ssnmf_input_label.shape[0]) if (ssnmf_input_label[x] == 0  and tot_labels[x] !=0)]
print (len(test_pos))
    

data_reconstruction_error = [ [0 for lambda_val in range(len(lambda_vals))] for lparam_val in range(len(lparam_vals)) ]     
label_reconstruction_error = [ [0 for lambda_val in range(len(lambda_vals))] for lparam_val in range(len(lparam_vals)) ]
evaluation = [ [0 for lambda_val in range(len(lambda_vals))] for lparam_val in range(len(lparam_vals)) ]

print ("--------Starting hyperparameter testing--------------")

 
if(os.path.exists(fullfile) == False):
    os.makedirs(fullfile)
f = fullfile + ('all_img_feat_ssnmf_tr14_te5.npz')


with open(os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_subsampled_new/' + "results.txt", "a+") as f_res:
    f_res.write("L_param\t")
    f_res.write("lambda\t")
    f_res.write("DRE\t")
    f_res.write("LRE\t")
    f_res.write("Eval\t\n")

for lnum,L_param in enumerate(lparam_vals):
    for inum,l in enumerate(lambda_vals):
        print ("--------On",lnum,inum,"--------------")
        if (L_param, l) in skip_list_lambda:
            print ("Skipping skip list value",L_param,l)
            continue
        feat_mat,label_mat,data_recon_err,label_recon_err,eval_s = ssnmf(ssnmf_input_data,ssnmf_input_label,rank,l,relax_label = True,L_param=L_param)
        data_reconstruction_error[lnum][inum] = data_recon_err
        label_reconstruction_error[lnum][inum] = label_recon_err
        evaluation[lnum][inum] = eval_s
        # f1sc = f1_score(label_mat, tot_labels)
        with open(os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_subsampled_new/' + "results.txt", "a+") as f_res:
            f_res.write(str(lparam_vals[lnum]) + "\t")
            f_res.write(str(lambda_vals[inum]) + "\t")
            f_res.write(str(data_reconstruction_error[lnum][inum]) + "\t")
            f_res.write(str(label_reconstruction_error[lnum][inum]) + "\t")
            f_res.write(str(evaluation[lnum][inum]) + "\t")
            f_res.write("\n----------------------------------------------\n")
        fname = fullfile + "label_feat_dump_" + str(L_param) + "_" + str(l) + ".npz"
        np.savez(fname,features=feat_mat,label_mat = label_mat,tot_labels=tot_labels, input_labels = ssnmf_input_label ,
                 test_pos = test_pos,data_recon_err = data_recon_err,label_recon_err=label_recon_err)    
        print ("L_param",L_param,"lambda",l,"DRE",data_recon_err,"LRE",label_recon_err,"Eval",eval_s)
        print("--- %s minutes ---" % ((time.time() - start_time)/60.)) 
