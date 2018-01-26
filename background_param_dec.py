import numpy
import theano
import matplotlib
matplotlib.use('Agg')
import os
import theano.tensor as T
from spectral import *
import spectral.io.envi as envi
from sklearn.preprocessing import normalize
from PIL import Image as pimg
import timeit
import cPickle
import matplotlib.pyplot as plt
start_time = timeit.default_timer()


def get_train_data(data, target):
    temp = range(target.shape[0])
    temp = numpy.array(temp)
    temp = temp*(target!=0)
    temp = temp.tolist()
    pos = [x for x in temp if x!=0]
    train_x = data[:,pos]
    train_y = target[pos]
    return [train_x,train_y]

def loaddata(file_no, bg_params, bg_param=0.02):
    img_te =[]
    gt_te  =[]
    img_full =[]
    cls = []
    class_d = []    
    
    for i in [file_no]:
        
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

        print img_te.shape
        
        temp1 = numpy.min(img_te[:,:226],axis=1)
        temp2 = numpy.max(img_te[:,:226],axis=1)
        temp1 = numpy.fabs(temp1)
        
        gt_te = img_te[:,226:]
        gt = numpy.zeros(gt_te.shape[0])
        gt1 = gt_te[:,0]*gt_te[:,1]+gt_te[:,1]*gt_te[:,2]+gt_te[:,2]*gt_te[:,0]
        gt1 = gt1 == 0
        gt = gt_te[:,0]* gt1 + gt_te[:,1]* gt1 +gt_te[:,2]* gt1

        
        gt = numpy.asarray(gt,dtype=int)        
        fullfile = os.path.dirname(os.path.abspath(__file__))+ '/bg_param/'
                 
        if(os.path.exists(fullfile) == False):
            os.makedirs(fullfile)
        temp4 =[]
        temp5 =[]
        for bg_param in bg_params: #[0.001, 0.01, 0.02,0.03,0.04, 0.05,0.06]:
            temp3 = (temp2 - temp1) < bg_param            
            # newImg1 = pimg.fromarray(numpy.uint8(255*(temp3.reshape([img.shape[0] ,img.shape[1]]))))
            # str4 = fullfile + ('bg_vs_fg%d.png'%(bg_param*1000))
            # newImg1.save(str4,"PNG")
            temp5 +=[bg_param]
            temp4 +=[numpy.mean(temp3*gt)*100.]
        fig, ax = plt.subplots()
        ax.set_ylim([-1, 5])
        ax.set_xlabel('Background Threshold')
        ax.set_ylabel('Misclassification Error (%)')
        ax.plot(temp5, temp4, marker='D', linestyle='-', color='r', markerfacecolor='b')
        ax.set_title( ("Error plot for image %d" % file_no) )
        plt.savefig( ("PAPER_bg_param_error/error-thres-%d.png" % file_no) )
        plt.close()
    return numpy.array(temp4)

    
if __name__ == '__main__':
    bg_params = [0.001, 0.01, 0.05, 0.07, 0.075, 0.08, 0.1, 0.12, 0.15, 0.17, 0.20, 0.25]
    files = range(1, 20)
    error = numpy.zeros(len(bg_params))
    for x in files:
        error += loaddata(x, bg_params)
    error /= len(files)
    
    fig, ax = plt.subplots()
    ax.set_ylim([-1, 5])
    ax.set_xlabel('Background Threshold')
    ax.set_ylabel('Misclassification Error (%)')
    ax.plot(bg_params, error, marker='D', linestyle='-', color='r', markerfacecolor='b')
    ax.set_title( ("Misclassification error for background threshold determination") )
    plt.savefig( ("PAPER_bg_param_error/error-thres-all.png") )
    plt.close()

    end_time = timeit.default_timer()

    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
