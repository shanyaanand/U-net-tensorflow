#from loaddata import loaddata, get_train_data
from collections import defaultdict
from ssnmf_func import ssnmf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from PIL import Image as pimg
import matplotlib
import os
from spectral import *
from sklearn import svm
from sklearn import metrics
import time
from load_ssnmf_feat import ssnmf_feat
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, centroid, weighted

def get_training_labels(data_labels_gt, clus_num, curcut_clus_labels):	
	cl_labels = data_labels_gt[(curcut_clus_labels == clus_num)]
	return cl_labels

start_time = time.time()

random_state = 170
color_val =spy_colors
color_val[0] = color_val[1]
color_val[1] = color_val[2]
color_val[2] = color_val[3]
color_val[3] = [0,0,0]
bg_param = 0.07

fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_tr14_te5/')
op_dir = fullfile + "/res_ssnmf_sshieclus/"
ranks = [4, 7, 10, 13, 16, 10]
lambda_lij_pair = [(0.01, 0.5)]

if(os.path.exists(op_dir) == False):
    os.makedirs(op_dir)
for rank in ranks:
	no_clus = 130 #KARTHIK: Modified after experiments with the combined dataset
	if rank == 10:
		ssnmf_feat_data = np.load(fullfile + "label_feat_dump_" + str(lambda_lij_pair[0][0]) + "_" + str(lambda_lij_pair[0][1]) + ".npz")
	else:
		ssnmf_feat_data = np.load(fullfile + "label_feat_dump_rank_" + str(rank) + "_" + str(lambda_lij_pair[0][0]) + "_" + str(lambda_lij_pair[0][1]) + ".npz")

	tot_feat = np.transpose(ssnmf_feat_data['features'])
	tot_feat = tot_feat [ ssnmf_feat_data['tot_labels'] != 0, :]

	n_top, o_top, test_pos = 0, 0, []
	while o_top < len(ssnmf_feat_data['tot_labels']):
		while ssnmf_feat_data['tot_labels'][o_top] == 0:
			o_top += 1
		if o_top == len(ssnmf_feat_data['tot_labels']):
			break
		if o_top in ssnmf_feat_data['test_pos']:
			test_pos.append(n_top)
		n_top += 1

	test_pos = np.array(test_pos)
	mask = np.ones(tot_feat.shape[0], dtype=bool)
	mask[test_pos] = False
	feat_mat, label_mat = (tot_feat[mask, :]), ssnmf_feat_data['tot_labels']
	label_mat = label_mat [ ssnmf_feat_data['tot_labels'] != 0 ]
	label_mat = label_mat[mask]

	# l_mask, t_mask = np.ones(label_mat.shape[0], dtype=bool), np.ones(ssnmf_feat_data['tot_labels'][~mask].shape[0], dtype=bool)
	# l_mask [ label_mat == 0 ] = False

	#te_mat, te_labels = np.transpose(tot_feat[:, 1-mask]), ssnmf_feat_data['tot_labels'][1-mask]
	print "feat mat shape",np.shape(feat_mat)
	rank = feat_mat.shape[1]
	print "Starting clustering"
	labels = np.array(KMeans(n_clusters=no_clus, random_state=random_state).fit_predict(tot_feat))
	print "Finished clustering", (time.time() - start_time),"secs"
	np.savez(op_dir + "cluster_labels_wo_unlabelled.npz", cluster_labels=labels)
	y_pred = np.empty((no_clus, tot_feat.shape[0]))
	y_pred[0] = labels.reshape((tot_feat.shape[0],))
	for j in range(no_clus):
		y_pred[j] = y_pred[0]

	p_count, pos = 0,[]

	for j in range(no_clus):
		temp_pos = [[x for x in range(tot_feat.shape[0]) if y_pred[0,x] == j]]
		pos += temp_pos
	redu_clus = np.empty((no_clus,rank))
	for j in range(no_clus):
		print "on",j,"with",len(pos[j])
		redu_clus[j] = np.mean(tot_feat[pos[j]],axis=0)
	Z = linkage(redu_clus,'ward')
	Z = np.asarray(Z,dtype = 'int32')

	for k in range(Z.shape[0]):
		tempr = min(Z[k,0], Z[k,1])
		temps = no_clus+k
		Z[k,2] = tempr
		for l in range(k+1,Z.shape[0]):
		        if Z[l,0] == temps:
		                Z[l,0] = tempr
		        if Z[l,1] == temps:
		                Z[l,1] = tempr
	for k in range(Z.shape[0]):
		mi_val = Z[k,2]
		clu_1 = Z[k,0]
		clu_2 = Z[k,1]
		pos[mi_val] = pos[clu_1]+pos[clu_2]
		y_pred[k+1,pos[mi_val]] = mi_val
		for j in range(k+2,no_clus):
		        y_pred[j] = y_pred[k+1]
	# np.savez(op_dir + "labels_after_cuts_" + str(rank) + "_" + str(no_clus) + ".npz", pred_labels=y_pred)
	# for k in range(no_clus):       
	# 	labels = y_pred[k].reshape(s0,s1) 		
	# 	f2 = (op_dir + 'hie_over_km_%d_%d.jpg'%(i,k))
	# 	save_rgb(f2, labels,colors=color_val )

	iternum, acc, prev, cur_cut, max_cut, max_acc = 0, 100, 0, Z.shape[0]-1, 0, 0
	assigned_labels = np.zeros_like(y_pred)

	while cur_cut >= 0: #(acc - prev) > 0.01 and 
		curcut_clus_labels = y_pred[cur_cut][ mask ]
		clus_num_assignment, class_samples, clus_purity = dict(), defaultdict(list), dict()
		prev = (acc if acc != 100 else 0)
		for clus_num in np.unique(curcut_clus_labels):
			cl_labels = get_training_labels(label_mat, clus_num, curcut_clus_labels)
			if len(cl_labels) == 0:
				for upper_cut in range(cur_cut + 1, Z.shape[0]):
					upper_clus_num = np.array(range(curcut_clus_labels.shape[0]))[curcut_clus_labels == clus_num][0]
					parent_clus_num = y_pred[upper_cut, upper_clus_num]
					_, pc = get_training_labels(label_mat, parent_clus_num, y_pred[upper_cut])
					if len(pc) != 0:
						clus_num_assignment[clus_num] = assigned_labels[upper_cut, upper_clus_num]
						break
			else:
				assigned_label = np.bincount(cl_labels).argmax()
				clus_num_assignment[clus_num] = assigned_label
				samples_classified = sum(cl_labels == assigned_label)
				class_samples[assigned_label].append( ( len(cl_labels), samples_classified / float(len(cl_labels)) ) )				
		for p in clus_num_assignment:
			assigned_labels[cur_cut, (y_pred[cur_cut] == p)] = clus_num_assignment[p]
		for n_class in class_samples:
			samples = [x[0] for x in class_samples[n_class]]
			accs = [x[1] for x in class_samples[n_class]]
			clus_purity[n_class] = 0
			for a in range(len(accs)):
				clus_purity[n_class] +=  (accs[a])*(float(samples[a])/sum(samples))
		acc = sum([v for k,v in clus_purity.iteritems()])/float(len(clus_purity))
		print np.histogram(assigned_labels[cur_cut, mask], bins=[0, 1, 2, 3, 4, 5, 6])
		print np.histogram(label_mat, bins=[0,1,2,3,4,5,6])
		print "@iteration",str(iternum),"cur_cut is",str(cur_cut),"weighted accuracy",str(acc),"diff",str(acc - prev)	
		f_s =  metrics.f1_score(assigned_labels[cur_cut, mask], label_mat, average='macro')
		print "tr_accuracy",f_s	
		print np.histogram(assigned_labels[cur_cut, ~mask], bins=[0, 1, 2, 3, 4, 5, 6])
		print np.histogram(ssnmf_feat_data['tot_labels'][~mask], bins=[0, 1, 2, 3, 4, 5, 6])
		fs_te = metrics.f1_score(assigned_labels[cur_cut, ~mask], ssnmf_feat_data['tot_labels'][~mask], average='macro')
		print "test_acc",fs_te
		with open("hieclus_results_cut_all.csv", "a+") as op:
			op.write(str(rank) + "," + str(cur_cut) + "," + str(acc) + "," + str(f_s) + "," + str(fs_te) + "\n")	
		cur_cut, iternum = cur_cut - 1, iternum + 1
		if acc > max_acc:
			max_cut, max_acc = cur_cut, acc
		if fs_te > f_sc:
			f_sc, f_cut = fs_te, cur_cut
	print("--- %s seconds ---" % (time.time() - start_time))
	start_time = time.time()
	print f_sc,"max test FScore"
	# opt_labels = assigned_labels[max_cut, :].reshape(s0, s1)
	# save_rgb(op_dir + "pred_img_" + str(i) + ".jpg", opt_labels, colors=color_val)