import numpy
import theano
import matplotlib
import os
import theano.tensor as T
from spectral import *
import spectral.io.envi as envi
from sklearn.preprocessing import normalize
from PIL import Image as pimg
import timeit
import cPickle
start_time = timeit.default_timer()


def dump_labels(bg_param=0.07):
    
    for i in range(1,21):
        print "Processing",i
        img_te =[]
        gt_te  =[]
        img_full =[]
        cls = []
        class_d = []    
        print i
        if i<11:
            str1=("Normal%d.hdr"%(i))
            str2=("Normal%d"%(i))
        else:
            i_new = (i%11)+1 
            str1=("Hyperplasia%d.hdr"%(i_new))
            str2=("Hyperplasia%d"%(i_new))            
        img=envi.open(str1,str2).load()
        img = numpy.asarray(img,dtype=theano.config.floatX)

        #print img.shape
        img_te = []

        for k in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_te.append(img[k,j,:])
        img_te = numpy.asarray(img_te,dtype=theano.config.floatX)
        
        gt_te = img_te[:,226:]
        gt = numpy.zeros(gt_te.shape[0])
        gt1 = gt_te[:,0]*gt_te[:,1]+gt_te[:,1]*gt_te[:,2]+gt_te[:,2]*gt_te[:,0]
        gt1 = gt1 == 0
        gt = gt_te[:,0]* gt1 + 2*gt_te[:,1]* gt1 + 3* gt_te[:,2]* gt1
        gt = numpy.asarray(gt,dtype=int)
        # gt = gt + 4*temp3

        if i<11:
            with open("normal_labels_wo_bg.csv", "a+") as f:
                f.write( ",".join([ str(x) for x in gt ]) + "\n" )
        else:
            with open("hyp_labels_wo_bg.csv", "a+") as f:
                f.write( ",".join([ str(x) for x in gt ]) + "\n" )

    return

dump_labels()