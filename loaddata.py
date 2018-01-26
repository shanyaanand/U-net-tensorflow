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
#import cPickle
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
            img = numpy.asarray(img,dtype=theano.config.floatX)

                    #print img.shape

            img_te = []
            for k in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img_te.append(img[k,j,:])

            img_te = numpy.asarray(img_te,dtype=theano.config.floatX)
            
            print ("shape",img_te.shape)
                    
            temp1 = numpy.min(img_te,axis=1)
            temp2 = numpy.max(img_te,axis=1)
            temp1 = numpy.fabs(temp1)
            if method == 'minmax':
                print ("Using min-max difference background rejection")
                temp3 = (temp2 - temp1) < bg_param
            elif method == 'energy':
                print ("Using spectral energy thresholding for background rejection")
                temp3 = numpy.linalg.norm(img_te, axis=1) < bg_param
            else:
                raise Exception("Unknown method for background rejection")



            print (numpy.mean(temp3==False))

           
        gt_te = img_te[:,767:]
        gt = numpy.zeros(gt_te.shape[0])
        gt1 = gt_te[:,0]*gt_te[:,1]+gt_te[:,1]*gt_te[:,2]+gt_te[:,2]*gt_te[:,0]
        gt1 = gt1 == 0
        
        gt = (gt_te[:,0]>0)* gt1 + 2*(gt_te[:,1]>0)* gt1 + 3* (gt_te[:,2]>0)* gt1
        

        gt = numpy.asarray(gt,dtype=int)
        gt = gt + 4*temp3
        
        
        


        print (sum(gt_te[:,0]*gt1  ==1),sum(gt_te[:,1]*gt1 == 1),sum(gt_te[:,2]*gt1 ==1))
        print (sum(gt==0),sum(gt==1),sum(gt==2),sum(gt==3),sum(gt==4))
                    
        data = numpy.transpose(img_te[:,:767])
        data = data*(data >= 0)
        rval = [data,gt,img.shape[0],img.shape[1]]
    print ('data shape' , data.shape)   
    return rval


def loaddata_spatial(file_no):
    img_te =[]
    gt_te  =[]
    img_full =[]
    cls = []
    class_d = []
    img_spa = []

    
    
    for i in [file_no]:
        
        print(i)
        str1=("Normal%d.hdr"%(i))
        str2=("Normal%d"%(i))
        img=envi.open(str1,str2).load()
        img = numpy.asarray(img,dtype=theano.config.floatX)

        #print img.shape
        img_te = []

        for k in range(2,img.shape[0]-2):
            for j in range(2,img.shape[1]-2):
                img_te.append(img[k,j,:])
                img_spa.append(img[k-2:k+3,j-2:j+3,:226].reshape([5*5*226]))

        img_te = numpy.asarray(img_te,dtype=theano.config.floatX)
        img_spa = numpy.asarray(img_spa,dtype=theano.config.floatX)

        print(img_te.shape)
        
        temp1 = numpy.min(img_te,axis=1)
        temp2 = numpy.max(img_te,axis=1)
        temp1 = numpy.fabs(temp1)
        temp3 = (temp2 - temp1) < 0.02

#        fullfile = os.path.dirname(os.path.abspath(__file__))+ '/res_ssnmf/' 
 
        
#        newImg1 = pimg.fromarray(numpy.uint8(255*(temp3.reshape([img.shape[0] ,img.shape[1]]))))
#        str4 = fullfile + ('bg_vs_fg%d.png'%(i))
#        newImg1.save(str4,"PNG")

        print(numpy.mean(temp3==False))

        
        gt_te = img_te[:,226:]
        gt = numpy.zeros(gt_te.shape[0])
        gt1 = gt_te[:,0]*gt_te[:,1]+gt_te[:,1]*gt_te[:,2]+gt_te[:,2]*gt_te[:,0]
        gt1 = gt1 == 0
        gt = gt_te[:,0]* gt1 + 2*gt_te[:,1]* gt1 + 3* gt_te[:,2]* gt1
        gt = numpy.asarray(gt,dtype=int)
        gt = gt + 4*temp3


        print (sum(gt_te[:,0]*gt1  ==1),sum(gt_te[:,1]*gt1 == 1),sum(gt_te[:,2]*gt1 ==1))
        print (sum(gt==0),sum(gt==1),sum(gt==2),sum(gt==3),sum(gt==4))
            
        data = numpy.transpose(img_spa)
        data = data*(data >= 0)
        rval = [data,gt,img.shape[0]-4,img.shape[1]-4]

    return rval


    
if __name__ == '__main__':

    c = loaddata(1)
    d = get_train_data(c[0], c[1])
    print(d[0].shape , d[1].shape)

    end_time = timeit.default_timer()

'''    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
'''