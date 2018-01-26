from sklearn import datasets
import numpy as np
from scipy.spatial import distance
import time
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, centroid, weighted


#fusermount -u /home/phani/1_BTP/clustering/remote
#sshfs amit-pc@172.16.21.21:/home/amit-pc/BTP_PK /home/phani/1_BTP/clustering/remote/


def ssnmf(data,target,rank,l,relax_label = True,L_param=0.001 ):

    X = data
    eval_s =True

    k = max(target)
    m = X.shape[0]
    n = X.shape[1]
    r = rank

    avg_X = X.mean()
    
    A = np.random.random(m * r).reshape(m, r) * avg_X
    S = np.random.random(r * n).reshape(r, n) * avg_X
    Y = np.zeros((k,n))
    B = np.ones((k,r))
    L = np.zeros((k,n))

    #print  X.shape, A.shape, S.shape, Y.shape, B.shape, L.shape

    temp = np.ones(k)

    for i in range(n):
        if target[i]!=0:
            Y[target[i]-1,i] = 1
            L[:,i] = temp
        if relax_label == True:
            L[target[i]-1,i] = L_param
    

    print  (X.shape, A.shape, S.shape, Y.shape, B.shape, L.shape)

    cost_func = 10000000
    cost_funcp = 1000000
    itera =0
    while(itera<100):

        cost_funcp = cost_func
        S_T = np.transpose(S)

        A = A*(np.dot(X,S_T)/(np.dot(np.dot(A,S),S_T)+0.000000001))
        B = B*(np.dot(L*Y,S_T)/(np.dot(L*np.dot(B,S),S_T)+0.000000001))
        A_T = np.transpose(A)
        B_T = np.transpose(B)
        S = S*((np.dot(A_T,X)+l*np.dot(B_T,L*Y))/(np.dot(A_T,np.dot(A,S))+l*np.dot(B_T,L*np.dot(B,S))+0.000000001))    

        data_recon_err = np.linalg.norm((X-np.dot(A,S)),ord ='fro')
        label_recon_err = np.linalg.norm(L*(Y-np.dot(B,S)),ord ='fro')
        cost_func = np.linalg.norm((X-np.dot(A,S)),ord ='fro') + l*(np.linalg.norm(L*(Y-np.dot(B,S)),ord ='fro')) # SSNMF cost funtion and here we need to vary 'l'--> lambda 
        print ('@iteration',str(itera),'cost is', str(cost_func),'\t','data and label reconstruction errors are', str(data_recon_err), 'and', str(label_recon_err),', respectively.')

        itera += 1
#        if(itera>30 and (l*label_recon_err/data_recon_err)>10):
#            eval_s = False
#            break
       
    label_matrix  =  np.dot(B,S)

    return S, label_matrix,data_recon_err,label_recon_err,eval_s

def gnmf(data,L,rank,l):

    X = data
    X_T = np.transpose(X)
    L_M = (np.absolute(L) - L)/2.
    L_P = (np.absolute(L) + L)/2.
    

    m = X.shape[0]
    n = X.shape[1]
    r = rank

    avg_X = X.mean()
    
    A = np.random.random(m * r).reshape(m, r) * avg_X
    S = np.random.random(r * n).reshape(r, n) * avg_X


    #print  X.shape, A.shape, S.shape, Y.shape, B.shape, L.shape

    cost_func = 10000000
    cost_funcp = 1000000
    itera =0
    while(itera<5):

        cost_funcp = cost_func
        S_T = np.transpose(S)

        A = A*(np.dot(X,S_T)/(np.dot(np.dot(A,S),S_T)+0.000000001))
        A_T = np.transpose(A)
        S_T = S_T*((np.dot(X_T,A)+(l*np.dot(L_M,S_T)))/(np.dot(np.dot(S_T,A_T),A)+(l*np.dot(L_P,S_T))+0.000000001))    

        S = np.transpose(S_T)

        data_recon_err = np.linalg.norm((X-np.dot(A,S)),ord ='fro')
        print(itera, 'data_re', data_recon_err)

        itera += 1
     

    return S,data_recon_err

def ssfnnmf(data,target,s0,s1,rank,l=1000,mu=0.01,relax_label = True):

    X = data

    k = max(target)
    m = X.shape[0]
    n = X.shape[1]
    r = rank
    assert(s0*s1 == n)

    avg_X = X.mean()
    A = np.random.random(m * r).reshape(m, r) * avg_X
    S = np.random.random(r * n).reshape(r, n) * avg_X
    Y = np.zeros((k,n))
    B = np.ones((k,r))
    L = np.zeros((k,n))
    SH = np.empty_like(S)
    SV = np.empty_like(S)
    temp = np.empty((s0,s1,r))
    temp1 = np.empty_like(temp)
    temp2 = np.empty_like(temp)



    #print  X.shape, A.shape, S.shape, Y.shape, B.shape, L.shape

    temp3 = np.ones(k)




    for i in range(n):
        if target[i]!=0:
            Y[target[i]-1,i] = 1
            L[:,i] = temp3
            if relax_label == True:
                L[target[i]-1,i] = 0.001
    

    print  (X.shape, A.shape, S.shape, Y.shape, B.shape, L.shape)




    cost_func = 10000000
    cost_funcp = 1000000
    itera =0
    while(itera<200):

        cost_funcp = cost_func
        
        S_T = np.transpose(S)
        temp = S_T.reshape((s0,s1,r))
        np.copyto(temp1,temp)
        temp1[range(s0-1),:,:] = temp[range(1,s0),:,:]
        SHT = temp1.reshape((n,r))
        SH = np.transpose(SHT)
        np.copyto(temp1,temp)
        temp1[:,range(s1-1),:] = temp[:,range(1,s1),:]
        SVT = temp1.reshape((n,r))
        SV = np.transpose(SVT)
        
        

        
        A = A*(np.dot(X,S_T)/(np.dot(np.dot(A,S),S_T)+0.000000001))
        B = B*(np.dot(L*Y,S_T)/(np.dot(L*np.dot(B,S),S_T)+0.000000001))
        A_T = np.transpose(A)
        B_T = np.transpose(B)
        S = S*((np.dot(A_T,X)+l*np.dot(B_T,L*Y)+ 2*mu*S)/(np.dot(A_T,np.dot(A,S))+l*np.dot(B_T,L*np.dot(B,S))+(mu*(SH+SV))+0.000000001))




        data_recon_err = np.linalg.norm((X-np.dot(A,S)),ord ='fro')
        label_recon_err = np.linalg.norm(L*(Y-np.dot(B,S)),ord ='fro')
        spatial_smoo_err= np.linalg.norm((2*S-SH-SV),ord ='fro')
        cost_func = np.linalg.norm((X-np.dot(A,S)),ord ='fro') + l*(np.linalg.norm(L*(Y-np.dot(B,S)),ord ='fro'))
        print (itera)
        print (cost_func,'\t','data_re', data_recon_err, 'label_re', label_recon_err,'smooth_re',spatial_smoo_err)

        itera += 1

        
    label_matrix  =  np.dot(B,S)

    return S

def ssgnmf(data,target,s0,s1,rank,relax_label = True):

    X = data

    k = max(target)
    m = X.shape[0]
    n = X.shape[1]
    r = rank
    l =100
    mu =0.1
    assert(s0*s1 == n)

    avg_X = X.mean()
    A = np.random.random(m * r).reshape(m, r) * avg_X
    S = np.random.random(r * n).reshape(r, n) * avg_X
    Y = np.zeros((k,n))
    B = np.ones((k,r))
    L = np.zeros((k,n))



    #print  X.shape, A.shape, S.shape, Y.shape, B.shape, L.shape

    temp3 = np.ones(k)




    for i in range(n):
        if target[i]!=0:
            Y[target[i]-1,i] = 1
            L[:,i] = temp3
            if relax_label == True:
                L[target[i]-1,i] = 0.001
    

    print (X.shape, A.shape, S.shape, Y.shape, B.shape, L.shape)




    cost_func = 10000000
    cost_funcp = 1000000
    itera =0
    while(itera<501):

        cost_funcp = cost_func
        
        S_T = np.transpose(S)        

        
        A = A*(np.dot(X,S_T)/(np.dot(np.dot(A,S),S_T)+0.000000001))
        B = B*(np.dot(L*Y,S_T)/(np.dot(L*np.dot(B,S),S_T)+0.000000001))
        A_T = np.transpose(A)
        B_T = np.transpose(B)
        S = S*((np.dot(A_T,X)+l*np.dot(B_T,L*Y))
               /(np.dot(A_T,np.dot(A,S))+l*np.dot(B_T,L*np.dot(B,S))+0.000000001))




        data_recon_err = np.linalg.norm((X-np.dot(A,S)),ord ='fro')
        label_recon_err = np.linalg.norm(L*(Y-np.dot(B,S)),ord ='fro')
        spatial_smoo_err= np.linalg.norm((2*S-SH-SV),ord ='fro')
        cost_func = np.linalg.norm((X-np.dot(A,S)),ord ='fro') + l*(np.linalg.norm(L*(Y-np.dot(B,S)),ord ='fro'))
        print (itera)

        print (cost_func,'\t','data_re', data_recon_err, 'label_re', label_recon_err,'smooth_re',spatial_smoo_err)

        itera += 1

        
    label_matrix  =  np.dot(B,S)

    return S


def nmf(data,rank):

    X = data

    m = X.shape[0]
    n = X.shape[1]
    r = rank

    avg_X = X.mean()
    A = np.random.random(m * r).reshape(m, r) * avg_X
    S = np.random.random(r * n).reshape(r, n) * avg_X



    cost_func = 10000000
    cost_funcp = 1000000
    itera =0
    #abs(cost_func-cost_funcp)>0.01
    while(itera<501):
        cost_funcp = cost_func
        S_T = np.transpose(S)
        A = A*((np.dot(X,S_T)/(np.dot(np.dot(A,S),S_T)+0.000000001)))
        A_T = np.transpose(A)
        S = S*((np.dot(A_T,X)/(np.dot(A_T,np.dot(A,S))+0.000000001)))    
        
        cost_func = np.linalg.norm((X-np.dot(A,S)),ord ='fro')

        print (itera, cost_func)
        itera+=1
        

        

    return S,cost_func




