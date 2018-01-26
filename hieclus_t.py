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
no_clus, random_state = 130, 170
ranks = [4, 7, 10, 13, 16, 10]
lambda_lij_pair = [(0.01, 0.5)]
fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/all_images_feat_ext_tr14_te5/')
op_dir = fullfile + "/res_ssnmf_sshieclus/"
print "Loading features"

for rank in ranks:
	ssnmf_feat_data = np.load(fullfile + "label_feat_dump_" + str(lambda_lij_pair[0][0]) + "_" + str(lambda_lij_pair[0][1]) + ".npz")
	tot_feat, test_pos = np.transpose(ssnmf_feat_data['features']), ssnmf_feat_data['test_pos']
	mask = np.ones(tot_feat.shape[0], dtype=bool)
	mask[test_pos] = False
	feat_mat, label_mat = (tot_feat[mask, :]), ssnmf_feat_data['tot_labels'][mask]
	l_mask, t_mask = np.ones(label_mat.shape[0], dtype=bool), np.ones(ssnmf_feat_data['tot_labels'][~mask].shape[0], dtype=bool)

	l_mask [ label_mat == 0 ] = False
	t_mask [ ssnmf_feat_data['tot_labels'][~mask] == 0 ] = False

	print np.histogram(ssnmf_feat_data['tot_labels'], bins=[0, 1, 2, 3, 4, 5, 6])
	print np.histogram(ssnmf_feat_data['tot_labels'][test_pos], bins=[0, 1, 2, 3, 4, 5, 6])
	print np.histogram(ssnmf_feat_data['tot_labels'][~mask], bins=[0, 1, 2, 3, 4, 5, 6])
	print np.histogram(label_mat, bins=[0, 1, 2, 3, 4, 5, 6])
	print np.histogram(label_mat[l_mask], bins=[0, 1, 2, 3, 4, 5, 6])

	print "Starting clustering"
	labels = np.array(KMeans(n_clusters=no_clus, random_state=random_state).fit_predict(tot_feat))
	print "Finished clustering", (time.time() - start_time),"secs"
	np.savez(op_dir + "cluster_labels_wo_unlabelled.npz", cluster_labels=labels)
	# print "Loading cluster labels"
	# labels = np.load(op_dir + "cluster_labels.npz")['cluster_labels']
	
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
		redu_clus[j] = np.mean(tot_feat[pos[j]],axis=0)
	print "Performing linkage"
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

	iternum, acc, prev, cur_cut, max_cut, max_acc, f_cut, f_sc = 0, 100, 0, Z.shape[0]-1, 0, 0, 0, 0
	assigned_labels = np.zeros_like(y_pred)

	print "Starting cuts"

	while cur_cut >= 0: #(acc - prev) > 0.01 and 
		
		curcut_clus_labels = y_pred[cur_cut][ mask ][ l_mask ]	
		clus_num_assignment, class_samples, clus_purity = dict(), defaultdict(list), dict()
		prev = (acc if acc != 100 else 0)
		# print "\n\nNumber of unique clusters",len(np.unique(curcut_clus_labels)),len(np.unique(y_pred[cur_cut][~mask]))
		# raw_input()
		for clus_num in np.unique(curcut_clus_labels):
			cl_labels = get_training_labels(label_mat[l_mask], clus_num, curcut_clus_labels)
			cl_labels = cl_labels[cl_labels != 0]
			# print "class labels for",clus_num,":",len(cl_labels)
			# print np.histogram(cl_labels, bins=[0, 1, 2, 3, 4, 5, 6])
			if len(cl_labels) == 0:
				# print "No class labels found...picking from parent"
				for upper_cut in range(cur_cut + 1, Z.shape[0]):
					upper_clus_num = np.array(range(curcut_clus_labels.shape[0]))[curcut_clus_labels == clus_num][0]
					parent_clus_num = y_pred[upper_cut, upper_clus_num]
					pc = get_training_labels(label_mat[l_mask], parent_clus_num, y_pred[upper_cut][mask][l_mask])
					if len(pc) != 0:
						clus_num_assignment[clus_num] = assigned_labels[upper_cut, upper_clus_num]
						break
			else:
				assigned_label = np.bincount(cl_labels).argmax()
				# print "\t",clus_num,assigned_label,np.histogram(cl_labels, bins=[0, 1, 2, 3, 4, 5, 6])
				clus_num_assignment[clus_num] = assigned_label
				samples_classified = sum(cl_labels == assigned_label)
				class_samples[assigned_label].append( ( len(cl_labels), samples_classified / float(len(cl_labels)) ) )				
		for p in clus_num_assignment:
			# print sum(y_pred[cur_cut][mask] == p), sum(y_pred[cur_cut][~mask] == p), assigned_labels[cur_cut, (y_pred[cur_cut] == p)].shape
			assigned_labels[cur_cut, (y_pred[cur_cut] == p)] = clus_num_assignment[p]
		# print np.histogram(assigned_labels[cur_cut, :], bins=[0, 1, 2, 3, 4, 5, 6])
		
		for n_class in class_samples:
			samples = [x[0] for x in class_samples[n_class]]
			accs = [x[1] for x in class_samples[n_class]]
			clus_purity[n_class] = 0
			for a in range(len(accs)):
				clus_purity[n_class] +=  (accs[a])*(float(samples[a])/sum(samples))
		acc = sum([v for k,v in clus_purity.iteritems()])/float(len(clus_purity))
		print np.histogram(assigned_labels[cur_cut, mask][l_mask], bins=[0, 1, 2, 3, 4, 5, 6])
		print np.histogram(label_mat[l_mask], bins=[0,1,2,3,4,5,6])
		print str(iterations),"@iteration",str(iternum),"cur_cut is",str(cur_cut),"weighted accuracy",str(acc),"diff",str(acc - prev)	
		f_s =  metrics.f1_score(assigned_labels[cur_cut, mask][ l_mask ], label_mat[l_mask], average='macro')
		print "tr_accuracy",f_s	
		print np.histogram(assigned_labels[cur_cut, ~mask][t_mask], bins=[0, 1, 2, 3, 4, 5, 6])
		print np.histogram(ssnmf_feat_data['tot_labels'][~mask][t_mask], bins=[0, 1, 2, 3, 4, 5, 6])
		fs_te = metrics.f1_score(assigned_labels[cur_cut, ~mask][t_mask], ssnmf_feat_data['tot_labels'][~mask][t_mask], average='macro')
		print "test_acc",fs_te
		with open("hieclus_results_new.csv", "a+") as op:
			op.write(str(cur_cut) + "," + str(acc) + "," + str(f_s) + "," + str(fs_te) + "\n")	
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