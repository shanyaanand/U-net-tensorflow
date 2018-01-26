from sklearn import datasets
import numpy as np
from scipy.spatial import distance
import time
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, centroid, weighted
from operator import itemgetter
from iter_projection import iter_projection,phi,const_mat
#from final_iter_pro_r import iter_projection,phi,const_mat
from tempfile import TemporaryFile

#fusermount -u /home/phani/1_BTP/clustering/remote
#sshfs amit-pc@172.16.21.21:/home/amit-pc/BTP_PK /home/phani/1_BTP/clustering/remote/

np.set_printoptions(precision=5, suppress=True)

start_time = time.time()

rng = np.random.RandomState(23455)

iris = datasets.load_iris()

perm = rng.permutation(iris.target.size)

no_labels = 50

no_samples = 150
data = iris.data[perm][:no_samples]
target = iris.target[perm][:no_samples]


# G is disimilarity matrix

G_flat = distance.pdist(data)
G = distance.squareform(G_flat)

print G.shape, data.shape, target.shape

#print iris.target




C = const_mat(target,no_samples,no_labels)
print C.shape


M= iter_projection(G_flat,C,no_samples)






def link_func(dist_mat):
    
    a = dist_mat.shape[0]
    D = distance.squareform(dist_mat,checks = False)
    list1 =list();
    for i in range(a):
        for j in range(a):
            if j>i:
                list1 += [[i,j, dist_mat[i,j],1,1]]
    #print list1[:10]
    list2 = sorted(list1,key = itemgetter(2))
    #print list2[:10]
    
    b = len(list2)
    Z = np.zeros((a-1,4))
    print Z.shape
    count = 0;
    for i in range(b):
        c = list2[i][0]
        d = list2[i][1]
        e =list2[i][3]+list2[i][4]
        f = a+count
        if c==d:
            continue
        
        Z[count] =[c,d,list2[i][2],e]
        for j in range(b):
            if list2[j][0] == c or list2[j][0] == d :
                list2[j][0] = f
                list2[j][3] = e
            if list2[j][1] == c or list2[j][1] == d :
                list2[j][1] = f 
                list2[j][4] = e

        count = count+1
    return Z






Z = linkage(M,'average')
#Z = link_func(distance.squareform(M))

f = file("tmp.bin","wb")
np.save(f,Z)
f.close()
                         

print("--- %s seconds ---" % (time.time() - start_time))

