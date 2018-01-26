from ssnmf_func import ssnmf, nmf, ssfnnmf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image as pimg
import matplotlib
import os
from sklearn import tree
from spectral import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import label_propagation

color_val =spy_colors'*'
color_val[0] = color_val[1]
color_val[1] = color_val[2]
color_val[2] = color_val[3]
color_val[3] = [0,0,0]

fullfile = os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_tr14_te5/' 
res_file = fullfile + "allcomp_te_results"
tr_resf = fullfile + "allcomp_tr_results"

def loadfeat(curr_nmf):#function for reading files and give out numbers of array
    f = fullfile +curr_nmf
    feat = np.load(f)
    mask = np.ones(feat['tot_labels'].shape[0], dtype=bool)
    mask [ feat['tot_labels'] == 0 ] = False
    mask [ feat['test_pos'] ] = False
    return feat['features'],feat['tot_labels'],np.where(mask)[0],feat['test_pos'], feat['data_recon_err']

start_time = time.time()
rng = np.random.RandomState(2345)

# print rank,"-------------------------------------------------"
#
# print "OUR MODEL - ALPHA = 0.5"
#
# for iter_val in range(1):
#
#     feat_mat,label_mat,tot_labels,train_pos,test_pos,l,L_param,data_recon_err,label_recon_err=loadfeat('all_img82_feat_%d.npz'%(iter_val))
#     act_labels = tot_labels[test_pos]
#     #print l,L_param,data_recon_err,label_recon_err
#     target_names = ['class 0', 'class 1', 'class 2','back_gr']
#     lp_in_labels =np.empty_like(tot_labels)
#     np.copyto(lp_in_labels,tot_labels)
#     lp_in_labels[test_pos] = 0
#     lp_in_labels -= 1
#
#     clf = label_propagation.LabelSpreading(kernel='knn', alpha=0.5)
#     clf.fit(np.transpose(feat_mat), lp_in_labels)
#     y_pred_lp = clf.predict(np.transpose(feat_mat[:,test_pos]))
#
#     pred_labels_lp = y_pred_lp + 1
#
#     pred_labels = pred_labels_lp[test_pos]
#
#     act_labels = tot_labels[test_pos]
#     print metrics.f1_score(act_labels, pred_labels, average='macro'),'\n'


    #print s0,s1,y_pred.shape
    # print np.histogram(tot_labels[test_pos],[0,1,2,3,4,5])
    # print np.histogram(pred_labels,[0,1,2,3,4,5])
    #
    #
    # print np.histogram(tot_labels,[0,1,2,3,4,5])
    # print np.histogram(pred_labels_lp,[0,1,2,3,4,5])
    # f_s =  metrics.f1_score(act_labels, pred_labels, average='macro')
    # c_mat =  metrics.confusion_matrix(act_labels, pred_labels)
    #
    # print 'testing error'
    # print f_s,'\n'
    # print c_mat,'\n'
    #
    # print 'train error '
    # print metrics.f1_score(tot_labels[train_pos], pred_labels_lp[train_pos], average='macro')
    # print metrics.confusion_matrix(tot_labels[train_pos], pred_labels_lp[train_pos])
    #
    #
    # f = fullfile + ('res_%d.npz'%(iter_val))
    # np.savez(f,f_score =f_s,confusion_matrix = c_mat)

ranks = [4, 7, 10, 13, 16]

with open(res_file, "a+") as op:
    op.write("rank,kmeans,rf,dt,gnb,knn,\n")
with open(tr_resf, "a+") as op:
    op.write("rank,rf,dt,gnb,knn,\n")


for rank in ranks:
    te_res, tr_res = "",""
    if rank == 10:#reading npz file
        feat_mat,tot_labels,train_pos,test_pos,data_recon_err=loadfeat('label_feat_dump_0.01_0.5.npz')
    else:
        feat_mat,tot_labels,train_pos,test_pos,data_recon_err=loadfeat('label_feat_dump_rank_' + str(rank) +'_0.01_0.5.npz')
    act_labels = tot_labels[test_pos]
    print rank,"-------------------------------------------------"

    print "K MEANS"

    for iter_val in range(1):
        #clustering features using kmean
        cl = KMeans(n_clusters=4, random_state=170)
        cl.fit(np.transpose(feat_mat))
        y_pred_lp = cl.predict(np.transpose(feat_mat[:,test_pos]))

        for x in range(len(y_pred_lp)):
            if y_pred_lp[x]== 0:
                y_pred_lp[x] = 4

            elif y_pred_lp[x]==4:
                y_pred_lp[x] = 0

        print np.histogram(tot_labels[test_pos],[0,1,2,3,4,5,6])
        print np.histogram(y_pred_lp,[0,1,2,3,4,5,6])
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        print 'testing acc'
        print f_s,'\n'
        te_res += str(f_s) + ','

    print rank,"-------------------------------------------------"

    print "Random Forest"
# def rf(feat_mat, train_pos, tot_labels, test_pos, clf):
    for iter_val in range(1):        
        clf_rf = RandomForestClassifier(n_estimators=2, n_jobs=10, max_depth=4)
        clf_rf.fit(np.transpose(feat_mat[:,train_pos]), tot_labels[train_pos])
        tr_pred = clf_rf.predict(np.transpose(feat_mat[:,train_pos]))
        y_pred_lp = clf_rf.predict(np.transpose(feat_mat[:,test_pos]))
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        tr_fs = metrics.f1_score(tot_labels[train_pos], tr_pred, average='macro')

        print 'testing acc'
        print f_s
        print 'train acc '
        print tr_fs
        
        te_res += str(f_s) + ','
        tr_res += str(tr_fs) + ','

    print rank,"-------------------------------------------------"

    print "Decision Tree"

    for iter_val in range(1):

        clf_dt = tree.DecisionTreeClassifier()
        clf_dt.fit(np.transpose(feat_mat[:,train_pos]), tot_labels[train_pos])
        tr_pred = clf_dt.predict(np.transpose(feat_mat[:,train_pos]))
        y_pred_lp = clf_dt.predict(np.transpose(feat_mat[:,test_pos]))
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        tr_fs = metrics.f1_score(tot_labels[train_pos], tr_pred, average='macro')

        print 'testing acc'
        print f_s
        print 'train acc '
        print tr_fs
        
        te_res += str(f_s) + ','
        tr_res += str(tr_fs) + ','

    print rank,"-------------------------------------------------"

    print "Gaussian Naive Bayes"

    for iter_val in range(1):

        clf_gn = GaussianNB()
        clf_gn.fit(np.transpose(feat_mat[:,train_pos]), tot_labels[train_pos])
        tr_pred = clf_gn.predict(np.transpose(feat_mat[:,train_pos]))
        y_pred_lp = clf_gn.predict(np.transpose(feat_mat[:,test_pos]))
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        tr_fs = metrics.f1_score(tot_labels[train_pos], tr_pred, average='macro')

        print 'testing acc'
        print f_s
        print 'train acc '
        print tr_fs
        
        te_res += str(f_s) + ','
        tr_res += str(tr_fs) + ','

    print rank,"-------------------------------------------------"

    print "KNN"
def rf(feat_mat, train_pos, tot_labels, test_pos, clf):
    for iter_val in range(1):
        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=10)
        clf_knn.fit(np.transpose(feat_mat[:,train_pos]), tot_labels[train_pos])
        tr_pred = clf_knn.predict(np.transpose(feat_mat[:,train_pos]))
        y_pred_lp = clf_knn.predict(np.transpose(feat_mat[:,test_pos]))
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        tr_fs = metrics.f1_score(tot_labels[train_pos], tr_pred, average='macro')

        print 'testing acc'
        print f_s
        print 'train acc '
        print tr_fs
        
        te_res += str(f_s) + ','
        tr_res += str(tr_fs) + ','

    with open(res_file + ".csv", "a+") as op:
        op.write(str(rank) + "," + te_res + "\n")
    with open(tr_resf + ".csv", "a+") as op:
        op.write(str(rank) + "," + tr_res + "\n")
    print("--- %s minutes ---" % ((time.time() - start_time)/60.))
    start_time = time.time()

for rank in ranks:
    te_res, tr_res = "",""
    feat_mat,tot_labels,train_pos,test_pos,data_recon_err=loadfeat('label_feat_dump_rank_' + str(rank) +'_0_0.npz')
    act_labels = tot_labels[test_pos]
    print rank,"-------------------------------------------------NMF"

    print "K MEANS"

    for iter_val in range(1):

        cl = KMeans(n_clusters=4, random_state=170)
        cl.fit(np.transpose(feat_mat))
        y_pred_lp = cl.predict(np.transpose(feat_mat[:,test_pos]))

        for x in range(len(y_pred_lp)):
            if y_pred_lp[x]== 0:
                y_pred_lp[x] = 4

            elif y_pred_lp[x]==4:
                y_pred_lp[x] = 0

        print np.histogram(tot_labels[test_pos],[0,1,2,3,4,5,6])
        print np.histogram(y_pred_lp,[0,1,2,3,4,5,6])
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        print 'testing acc'
        print f_s,'\n'
        te_res += str(f_s) + ','

    print rank,"-------------------------------------------------NMF"

    print "Random Forest"

    for iter_val in range(1):
        clf_rf = RandomForestClassifier(n_estimators=5, n_jobs=10, max_depth=10)
        clf_rf.fit(np.transpose(feat_mat[:,train_pos]), tot_labels[train_pos])
        tr_pred = clf_rf.predict(np.transpose(feat_mat[:,train_pos]))
        y_pred_lp = clf_rf.predict(np.transpose(feat_mat[:,test_pos]))
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        tr_fs = metrics.f1_score(tot_labels[train_pos], tr_pred, average='macro')

        print 'testing acc'
        print f_s
        print 'train acc '
        print tr_fs
        
        te_res += str(f_s) + ','
        tr_res += str(tr_fs) + ','

    print rank,"-------------------------------------------------NMF"

    print "Decision Tree"

    for iter_val in range(1):

        clf_dt = tree.DecisionTreeClassifier()
        clf_dt.fit(np.transpose(feat_mat[:,train_pos]), tot_labels[train_pos])
        tr_pred = clf_dt.predict(np.transpose(feat_mat[:,train_pos]))
        y_pred_lp = clf_dt.predict(np.transpose(feat_mat[:,test_pos]))
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        tr_fs = metrics.f1_score(tot_labels[train_pos], tr_pred, average='macro')

        print 'testing acc'
        print f_s
        print 'train acc '
        print tr_fs
        
        te_res += str(f_s) + ','
        tr_res += str(tr_fs) + ','

    print rank,"-------------------------------------------------NMF"

    print "Gaussian Naive Bayes"

    for iter_val in range(1):

        clf_gn = GaussianNB()
        clf_gn.fit(np.transpose(feat_mat[:,train_pos]), tot_labels[train_pos])
        tr_pred = clf_gn.predict(np.transpose(feat_mat[:,train_pos]))
        y_pred_lp = clf_gn.predict(np.transpose(feat_mat[:,test_pos]))
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        tr_fs = metrics.f1_score(tot_labels[train_pos], tr_pred, average='macro')

        print 'testing acc'
        print f_s
        print 'train acc '
        print tr_fs
        
        te_res += str(f_s) + ','
        tr_res += str(tr_fs) + ','

    print rank,"-------------------------------------------------NMF"

    print "KNN"

    for iter_val in range(1):

        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=10)
        clf_knn.fit(np.transpose(feat_mat[:,train_pos]), tot_labels[train_pos])
        tr_pred = clf_knn.predict(np.transpose(feat_mat[:,train_pos]))
        y_pred_lp = clf_knn.predict(np.transpose(feat_mat[:,test_pos]))
        f_s = metrics.f1_score(act_labels, y_pred_lp, average='macro'),'\n'
        tr_fs = metrics.f1_score(tot_labels[train_pos], tr_pred, average='macro')

        print 'testing acc'
        print f_s
        print 'train acc '
        print tr_fs
        
        te_res += str(f_s) + ','
        tr_res += str(tr_fs) + ','

    with open(res_file + "_nmf.csv", "a+") as op:
        op.write(str(rank) + "," + te_res + "\n")
    with open(tr_resf + "_nmf.csv", "a+") as op:
        op.write(str(rank) + "," + tr_res + "\n")
    print("--- %s minutes ---" % ((time.time() - start_time)/60.))
    start_time = time.time()


for rank in ranks:
    print rank,"-------------------------------------------------"
    if rank == 10:
        feat_mat,tot_labels,train_pos,test_pos,data_recon_err=loadfeat('label_feat_dump_0.01_0.5.npz')
    else:
        feat_mat,tot_labels,train_pos,test_pos,data_recon_err=loadfeat('label_feat_dump_rank_' + str(rank) +'_0.01_0.5.npz')
    act_labels = tot_labels[test_pos]

    print "SVM"
    for iter_val in range(1):

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
        
        str(f_s)
        str(tr_fs)
    with open(res_file + ".csv", "a+") as op:
        op.write(str(rank) + "," + str(f_s) + "\n")
    with open(tr_resf + ".csv", "a+") as op:
        op.write(str(rank) + "," + str(tr_fs) + "\n")
    print("--- %s minutes ---" % ((time.time() - start_time)/60.))
    start_time = time.time()

    print rank,"-------------------------------------------------NMF"
    feat_mat,tot_labels,train_pos,test_pos,data_recon_err=loadfeat('label_feat_dump_rank_' + str(rank) +'_0_0.npz')
    act_labels = tot_labels[test_pos]

    print "SVM"
    for iter_val in range(1):

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
        
        str(f_s)
        str(tr_fs)
    with open(res_file + "_nmf.csv", "a+") as op:
        op.write(str(rank) + "," + str(f_s) + "\n")
    with open(tr_resf + "_nmf.csv", "a+") as op:
        op.write(str(rank) + "," + str(tr_fs) + "\n")
    print("--- %s minutes ---" % ((time.time() - start_time)/60.))
    start_time = time.time()
