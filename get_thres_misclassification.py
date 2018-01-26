import numpy as np
import theano
from spectral import *
import spectral.io.envi as envi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def misclassification_count(file_no,bg_param=0.07):
    img_te =[]
    gt_te  =[]
    img_full =[]
    cls = []
    class_d = []
    for i in [file_no]:
        if i<11:
            str1=("Normal%d.hdr"%(i))
            str2=("Normal%d"%(i))

        else:
            i_new = (i%11)+1 
            str1=("Hyperplasia%d.hdr"%(i_new))
            str2=("Hyperplasia%d"%(i_new))            
        img=envi.open(str1,str2).load()
        img = np.asarray(img,dtype=theano.config.floatX)
        img_te = []

        for k in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_te.append(img[k,j,:])

        img_te = np.asarray(img_te,dtype=theano.config.floatX)

        temp1 = np.min(img_te,axis=1)
        temp2 = np.max(img_te,axis=1)
        temp1 = np.fabs(temp1)
        # t = np.linalg.norm(img_te, axis=1)
        temp3 = np.array((temp2 - temp1) < bg_param, dtype=int)
        # temp3 = np.array(t < bg_param, dtype=int)
        # print temp3, np.min(temp3), np.max(temp3), np.sum(temp3), temp3.shape, np.dot( np.ones(temp3.shape[0]), temp3 )
        
        gt_te = img_te[:,226:]
        gt = np.zeros(gt_te.shape[0])
        gt1 = gt_te[:,0]*gt_te[:,1]+gt_te[:,1]*gt_te[:,2]+gt_te[:,2]*gt_te[:,0]
        gt1 = gt1 == 0
        gt = gt_te[:,0]* gt1 + gt_te[:,1]* gt1 + gt_te[:,2]* gt1

        # print gt, np.min(gt), np.max(gt), np.sum(gt), gt.shape, np.dot( gt, temp3 )
        # print np.dot( gt, temp3 ), np.sum(gt * temp3)
        # raw_input()
	

	print str(bg_param),"Misclassified", str(np.mean( gt * temp3 )),"%" #, str(np.dot( np.ones(temp3.shape[0]), temp3 ))

	return np.dot(gt, temp3), np.dot( np.ones(temp3.shape[0]), temp3)

if __name__ == "__main__":
	bg_params = np.linspace(0., 1., 25)
	for img in range(1, 10):
		print "IMG",img
		fig, ax = plt.subplots()
		errors = []
		#fig.suptitle( ("Background masks for Normal %d" % img) )
		for _ix, x in enumerate(bg_params):
			mc, tot = misclassification_count(img, x)
			errors.append( float(mc)/float(tot) if tot != 0 else 0. )
		ax.plot(bg_params, errors)
		ax.set_title( ("Error plot for Normal %d" % img) )
		plt.savefig( ("error-thres-%d.png" % img) )
        plt.close()