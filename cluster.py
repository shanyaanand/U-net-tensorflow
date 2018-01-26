import numpy as np
import matplotlib.pyplot as plt 
import os
from spectral import *
import theano
import spectral.io.envi as envi
from numpy.linalg import inv
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans  


def loaddata(file_no,bg_param=0.07, method='minmax'):
	img_te =[]
	
	gt_te  =[]
	img_full =[]
	cls = []
	class_d = []

	for i in [file_no]:
		        
		print (i)


		if i<11:
			str1=("Normal %d.hdr"%(i))
			str2=("Normal %d"%(i))

		else:
			i_new = (i%11)+1 
			str1=("Hyperplasia %d.hdr"%(i_new))
			str2=("Hyperplasia %d"%(i_new))            

			img = envi.open(str1,str2).load()
		img = np.asarray(img,dtype=theano.config.floatX)

		        #print img.shape

		img_te = []
		for k in range(img.shape[0]):
			for j in range(img.shape[1]):
				img_te.append(img[k,j,:])

		img_te = np.asarray(img_te,dtype=theano.config.floatX)
		
		print ("shape",img_te.shape)
		        
		temp1 = np.min(img_te,axis=1)
		temp2 = np.max(img_te,axis=1)
		temp1 = np.fabs(temp1)
		if method == 'minmax':
			print ("Using min-max difference background rejection")
			temp3 = (temp2 - temp1) < bg_param
		elif method == 'energy':
			print ("Using spectral energy thresholding for background rejection")
			temp3 = np.linalg.norm(img_te, axis=1) < bg_param
		else:
			raise Exception("Unknown method for background rejection")



		print (np.mean(temp3==False))

		   
		gt_te = img_te[:,767:]
		gt = np.zeros(gt_te.shape[0])
		gt1 = gt_te[:,0]*gt_te[:,1]+gt_te[:,1]*gt_te[:,2]+gt_te[:,2]*gt_te[:,0]
		gt1 = gt1 == 0
		
		gt = (gt_te[:,0]>0)* gt1 + 2*(gt_te[:,1]>0)* gt1 + 3* (gt_te[:,2]>0)* gt1
		

		gt = np.asarray(gt,dtype=int)
		gt = gt + 4*temp3
		
		
		


		print (sum(gt_te[:,0]*gt1  ==1),sum(gt_te[:,1]*gt1 == 1),sum(gt_te[:,2]*gt1 ==1))
		print (sum(gt==0),sum(gt==1),sum(gt==2),sum(gt==3),sum(gt==4))
		            
		data = np.transpose(img_te[:,:767])
		data = data*(data >= 0)
		rval = [data,gt,img.shape[0],img.shape[1]]
	print ('data shape' , data.shape)	
	return rval
  

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
        data_train = np.hstack((data_train,data_oi[:,shuff_labeled_pos]))
        label_train = np.hstack((label_train,tot_labels[shuff_labeled_pos]))

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


