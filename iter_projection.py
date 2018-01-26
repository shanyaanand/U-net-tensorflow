import numpy as np
import time
from scipy.spatial import distance
import os
import sys
from scipy import optimize


start_time = time.time()



def const_mat(target,nos,nol):
    m = nos*(nos-1)
    C = []
    temp = [0 for x in range(m/2)] 
    for i in range(nol):
        tmp_label1 = target[i]
        for j in range(nol):
            tmp_label2 = target[j]
            for k in range(nol):
                tmp_label3 = target[k]
                if tmp_label1 == tmp_label2 and tmp_label1 != tmp_label3 and i!=j:
                    j_telda = j-1-i
                    for l in range(i):
                        j_telda = j_telda + (nos-1-l)
                    temp[j_telda] = 1

                    k_telda = k-1-i
                    for l in range(i):
                        k_telda = k_telda + (nos-1-l)
                    temp[k_telda] = -1
                    C = C + [temp]
                    temp = [0 for x in range(m/2)] 
    C =np.array(C)
    return C
            




def phi(a,*args):
#    print("--- %s seconds ---" % (time.time() - start_time))
    d,t,n,q,r,C = args
    A = distance.squareform(a)
#    temp4 = a-d 
    
    L = np.sum((a-d)**2)   
#    L = np.square(np.linalg.norm(a - d))
#    temp1 = []
    
    temp = np.sum(C*a,axis =1)
    temp= (temp>0)*temp
    T = np.sum(temp**2)

    
#    T = np.square(np.linalg.norm(temp))

#    for i in np.dot(C,a):
#        if i>0:
#            T = T + np.square(i)
#    print A.shape
#    P=0
#    for i in range(n):
#        for j in range(n):

#            temp3 = np.min(A[[i,j],:],axis =0)
#            P = P + np.sum(((temp3>A[i,j])*(A[i,:]-A[j,:]))**2)
            
#    temp5  = np.append(r*temp,temp4)
#    temp6 = np.append(temp5,q*temp2)

#    phi_out = temp6  
#    phi_out = L 
#    print("--- %s seconds ---" % (time.time() - start_time))
    return L +q*T

def phi_grad(a,*args):
    return a
    


def iter_projection(d, C, n):
    m =n*(n-1)/2
    assert(d.size == m),"disimilarity matrix dimension mismatch"
    a = np.empty_like(d)
    rng = np.random.RandomState(23455)
    d_telda = np.mean(d)
    print d_telda
    sd = (2./3.)*(np.square(np.linalg.norm(d-d_telda)))/(n*(n-1))
    print sd 
    a = d + rng.normal(loc=0,scale = sd,size=m)
    print a.shape, d.shape, rng.normal(loc=0,scale = sd,size=m).shape
    
    a_prev = np.zeros_like(d)
    r = C.shape[0]
    u = np.zeros(r)
    u_prev = np.zeros_like(u)
    E = np.identity(m)
    t =1
    L = np.square(np.linalg.norm(a - d))
    P = 0
    A = distance.squareform(a)
    temp = np.dot(C,a)
    temp= (temp>0)*temp
    T = np.square(np.linalg.norm(temp))
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if A[i,j] < min(A[i,k],A[j,k]):
                    P = P + np.square(A[i,k]-A[j,k])

    
#    while(np.sum((a_prev-a)**2)>1 and t<20):
    while(t<10):
        if 1==1:
            np.copyto(a_prev,a)
            np.copyto(u_prev,u)
            
                    
            if t ==1:
                q = L/P
                r = L/T
            else:
                q = 10*q
                r = 10*r
                
#            print("--- %s seconds ---" % (time.time() - start_time))
            a = optimize.fmin_l_bfgs_b(phi,a_prev,args=(d,t,n,q,r,C),approx_grad =True)
            print("--- %s seconds ---\n" % (time.time() - start_time))
            a = a[0]
            temp = np.dot(C,a)
            temp= (temp>0)*temp
            T = np.sum(temp**2)
            A = distance.squareform(a)
            temp1 =[]
            for i in range(n):
                for j in range(n):                    
                    for k in range(n):
                        if A[i,j] < min(A[i,k],A[j,k]):
                            temp1 = temp1 + [A[i,k]-A[j,k]]

            temp2 = np.array(temp1)
            P = np.sum(temp2**2)
#            print a
            print 'dista err\t', np.sum((a-d)**2)
            print 'label err\t', T
            print 'u met err\t', P,'\n'

            t =t+1
            
    return a

if __name__ == '__main__':        
        
    start_time = time.time()

    G = np.array([[0,1,7,9,11,11],
               [1,0,7,9,12,12],
               [7,7,0,6,10,10],
               [9,9,6,0,8,8],
               [11,12,10,8,0,2],
               [11,12,10,8,2,0]])
    d = distance.squareform(G)

    C = np.array([[1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
               [0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0,0,-1,0,0,0,0,0],
                  [0,0,-1,0,0,0,0,0,0,0,0,0,1,0,0],])
    
    out = iter_projection(d,C,6)
    print distance.squareform(out)
    
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((time.time() - start_time) / 60.))
