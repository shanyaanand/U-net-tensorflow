from loaddata import loaddata, get_train_data
from ssnmf_func import ssnmf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image as pimg
import matplotlib
import os
from spectral import *
from sklearn import svm
from sklearn.model_selection import KFold
import time
from load_ssnmf_feat import ssnmf_feat


def loadfeat(curr_nmf):    
    fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_tr14_te5/') 
    f = fullfile + curr_nmf
    feat = np.load(f)
    return feat['features'],feat['label_mat'],feat['tot_labels'],feat['test_pos'],\
           feat['data_recon_err'],feat['label_recon_err']

start_time = time.time()

random_state = 170 # For reproducibility

lambda_vals = [0.5] # 1, 2, 0.1, 3, 10]         # To see the variations in regulartization parameter lambda
lparam_vals = [0.001, 0.005, 0.01, 0.0001] # To see the variations in relaxation parameter L_ij
ncluster_vals = [60, 70, 80, 90, 100, 110, 120, 130]
skip_list = [0.001]
fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_tr14_te5/') 

VALIDATION_SPLIT = 15

op_file = fullfile + "cluster_validation_0.5.csv"
rank_file = fullfile + "cluster_rank_0.5.csv"
time_file = fullfile + "cluster_time_0.5.csv"

with open(op_file, "a+") as op:
	op.write("L_ij,lambda,ncluster,validation_iteration,validation_error\n")
with open(rank_file, "a+") as rank:
	rank.write("L_ij,lambda,ncluster,avg_error\n")
with open(time_file, "a+") as time_f:
	time_f.write("L_ij,lambda,ncluster,validation_iteration,time(secs)\n")

for lnum,L_param in enumerate(lparam_vals):
    if L_param in skip_list:
    	print "Skipping value in skip_list"
    	continue
    for inum,l in enumerate(lambda_vals):
        print "--------On",L_param,l,"--------------"
        feat_mat,_,label_mat,test_pos,_,_=loadfeat('label_feat_dump_' + str(L_param) + '_' + str(l) + '.npz')
        mask = np.ones(label_mat.shape, dtype=bool)     
        mask[test_pos] = False
        train_mat, train_labels = np.transpose(feat_mat[:, mask]), np.transpose(label_mat[mask])
        print "Loaded training data and labels"
        k_split = KFold(n_splits=VALIDATION_SPLIT)
        for ncluster in ncluster_vals:
    		print "Performing KMeans and label fitting"
    		y_pred = np.array(KMeans(n_clusters=ncluster, random_state=random_state).fit_predict( train_mat ))
    		print np.unique(y_pred), np.bincount(y_pred)
    		validation_num, validation_errors = 0, []
    		print "Fitting done...performing validation"
    		for train, validation in k_split.split(train_mat):
    			print np.shape(train), np.shape(validation)
    			tr_b, val_b = np.zeros(train_labels.shape[0], dtype=bool), np.zeros(train_labels.shape[0], dtype=bool)
    			tr_b[train], val_b[validation] = True, True
    			samples_misclassified, tot_samples = 0, 0
    			for c_num in range(ncluster):
    				print "On cluster",c_num
    				tr_l, val_l = (train_labels[ tr_b & (y_pred == c_num) ]), (train_labels[ val_b & (y_pred == c_num) ])    				
    				tr_l, val_l = tr_l[tr_l != 0], val_l[val_l != 0]
    				print np.shape(tr_l), np.shape(val_l)
    				if tr_l.shape[0] == 0 :
    					print "Skipping",c_num,"due to lack of samples in training"
    					continue
    				assigned_label = np.bincount(tr_l).argmax()
    				samples_misclassified += sum(val_l != assigned_label)
    				tot_samples += len(val_l)
    			validation_errors.append ( float(samples_misclassified)/tot_samples )
    			validation_num += 1
    			with open(op_file, "a+") as op:
    				op.write(str(L_param) + "," + str(l) + "," + str(ncluster) + "," + str(validation_num) + "," + str(validation_errors[-1]) + "\n")
    			with open(time_file, "a+") as time_f:
					time_f.write(str(L_param) + "," + str(l) + "," + str(ncluster) + "," + str(validation_num) + "," + str((time.time() - start_time)/60.) + "\n")
    			print "------------", str(validation_num), ":", str((time.time() - start_time)/60.), "mins --------------------"
    			start_time = time.time()
    		with open(rank_file, "a+") as rank:
    			rank.write(str(L_param) + "," + str(l) + "," + str(ncluster) + "," + str( float(sum(validation_errors))/len(validation_errors) ) + "\n" )