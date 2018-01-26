
'''
U-net implementation in Tesnsorflow

Objective: segmentation of images into four class(epithelium,goblet,stroma, and background)

y = f(X)

X: image (None,None,25)
Y: mask (None,None,4)
	-epithelium is class 1
	-goblet is class 2
	-stroma is class 3
	-background is class 4
Loss function: maximize IOU

	(intersection of prediction & grount truth)
	-------------------------------
	(union of prediction & ground truth)

Notes: 
	In the paper,the pixel-wise softmax was used
	but, I used the IOU and pixel-wise sigmoid 

Original Paper:
	https://arxiv.org/abs/1505.04597

'''



import tensorflow as tf 
import numpy as np 
import os
import matplotlib.pyplot as plt 
import random

X = tf.placeholder(tf.float32,shape=[None,None,None,25])
Y = tf.placeholder(tf.float32,shape=[None,None,None,4])
lamda = 1e-2
acc_trainig = []
acc_test = []
n_epoch = 5

def IOU_(y_true, y_pred):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """

    pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_true)[1]*tf.shape(y_true)[2]])
    true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[1]*tf.shape(y_true)[2]])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)




def mirror(arr,img):

"""
Args:
	input_ (1-D array and 3-D array):([extension_in_right_side,extension_in_left_side,extension_in_Upper_side,extension_in_down_side]) and ([Height,Width,Depth])

Return: 
	img: Output extended image

"""



	for start,end in enumerate(arr):
		

		if start == 0:		
			temp=img[:,:end,:]
			
			img = np.concatenate([np.flip(temp,axis = 1),img],axis = 1)
			
		if start == 1 :
			temp=img[:,start*img.shape[1]-end:img.shape[1],:]
			img = np.concatenate([img,np.flip(temp,axis=1)],axis=1)
			
		if start == 2:
			
			temp=img[:end,:,:]			
						
			img=np.concatenate([np.flip(temp,axis=0),img],axis=0)
			
		if start == 3:
			temp=img[img.shape[0]-end:img.shape[0],:,:]
			
			img=np.concatenate([img,np.flip(temp,axis=0)],axis = 0)	
		
	return img


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, shape_wieght):
	W = weight_variable(shape_wieght)
	B = bias_variable([shape_wieght[3]])
	h_conv_out = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + B)
	mean, variance = tf.nn.moments(h_conv_out, [0, 1, 2])
	beta = tf.Variable(dtype=tf.float32,initial_value=np.ones(1))
	gamma = tf.Variable(dtype=tf.float32,initial_value=np.ones(1))
	epsilon = tf.constant(1e-8)
	scale_after_norm = tf.Variable(1)
	return tf.nn.batch_norm_with_global_normalization(h_conv_out, mean, variance, beta, gamma,epsilon, scale_after_norm)


def de_conv2d(x,shape_wieght,output_shape):
	W = weight_variable(shape_wieght)
	B = bias_variable([shape_wieght[2]])
	return tf.nn.conv2d_transpose(x, W,output_shape = output_shape ,strides=[1, 2, 2, 1], padding='SAME') + B

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


#convo layer 1 
h_conv1 = conv2d(X,[3,3,25,64])

#convo layer 2  
h_conv2 = conv2d(h_conv1,[3,3,64,64])

#max_pool 1  stack 2
h_max1 = max_pool_2x2(h_conv2)

#convo layer 3
h_conv3 = conv2d(h_max1,[3,3,64,128])

#convo layer 4 
h_conv4 = conv2d(h_conv3,[3,3,128,128])

#max_pool 2

h_max2 = max_pool_2x2(h_conv4)

#convo layer 5 

h_conv5 = conv2d(h_max2,[3,3,128,256])

#convo layer 6

h_conv6 = conv2d(h_conv5,[3,3,256,256])

#max_pool 3

h_max3 = max_pool_2x2(h_conv6)

#convo layer 7 

h_conv7 = conv2d(h_max3,[3,3,256,512])

#convo layer 8

h_conv8 = conv2d(h_conv7,[3,3,512,512])


#upconvo layer 1

output_batch_size = tf.shape(h_conv8)[0] 
output_row = tf.shape(h_conv8)[1]*2
output_column = tf.shape(h_conv8)[2]*2
output_shape = [output_batch_size,output_row,output_column,256]
h_upconv1 = de_conv2d(h_conv8,[2,2,256,512],output_shape)

#convo layer 9

size_row = tf.shape(h_upconv1)[1]
size_column = tf.shape(h_upconv1)[2]
h_cropped_prev_conv1 = tf.image.resize_nearest_neighbor(h_conv6,(size_row,size_column))
h_conv9 = tf.concat([h_upconv1,h_cropped_prev_conv1],3)

#convo layer 10

h_conv10 = conv2d(h_conv9,[3,3,512,256])

#convo layer 11

h_conv11 = conv2d(h_conv10,[3,3,256,256])

#upconvo layer 2 
output_batch_size = tf.shape(h_conv11)[0] 
output_row = tf.shape(h_conv11)[1]*2
output_column = tf.shape(h_conv11)[2]*2
output_shape = [output_batch_size,output_row,output_column,128]
h_upconv2 = de_conv2d(h_conv11,[2,2,128,256],output_shape)

#convo layer 12

size_row = tf.shape(h_upconv2)[1]
size_column = tf.shape(h_upconv2)[2]
h_cropped_prev_conv2 = tf.image.resize_nearest_neighbor(h_conv4,(size_row,size_column))
h_conv12 = tf.concat([h_upconv2,h_cropped_prev_conv2],3)

#convo layer 13

h_conv13 = conv2d(h_conv12,[3,3,256,128])

#convo layer 14

h_conv14 = conv2d(h_conv13,[3,3,128,128])

#upconvo layer 3 

output_batch_size = tf.shape(h_conv14)[0] 
output_row = tf.shape(h_conv14)[1]*2
output_column = tf.shape(h_conv14)[2]*2
output_shape = [output_batch_size,output_row,output_column,64]
h_upconv3 = de_conv2d(h_conv14,[2,2,64,128],output_shape)

#convo layer 15

size_row = tf.shape(h_upconv3)[1]
size_column = tf.shape(h_upconv3)[2]
h_cropped_prev_conv3 = tf.image.resize_nearest_neighbor(h_conv2,(size_row,size_column))
h_conv15 = tf.concat([h_upconv3,h_cropped_prev_conv3],3)

#convo layer 16

h_conv16 = conv2d(h_conv15,[3,3,128,64])

#convo layer 17(Output Layer)

Y_pred = tf.nn.sigmoid(conv2d(h_conv16,[1,1,64,4]))

correct_prediction = tf.equal(tf.argmax(Y_pred, 3),tf.argmax(Y, 3))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	

IOU1 = IOU_(tf.slice(Y,[0,0,0,0],[tf.shape(Y)[0],tf.shape(Y)[1],tf.shape(Y)[2],1]),tf.slice(Y_pred,[0,0,0,0],[tf.shape(Y_pred)[0],tf.shape(Y_pred)[1],tf.shape(Y_pred)[2],1]))
IOU2 = IOU_(tf.slice(Y,[0,0,0,1],[tf.shape(Y)[0],tf.shape(Y)[1],tf.shape(Y)[2],1]),tf.slice(Y_pred,[0,0,0,1],[tf.shape(Y_pred)[0],tf.shape(Y_pred)[1],tf.shape(Y_pred)[2],1]))
IOU3 = IOU_(tf.slice(Y,[0,0,0,2],[tf.shape(Y)[0],tf.shape(Y)[1],tf.shape(Y)[2],1]),tf.slice(Y_pred,[0,0,0,2],[tf.shape(Y_pred)[0],tf.shape(Y_pred)[1],tf.shape(Y_pred)[2],1]))
IOU4 = IOU_(tf.slice(Y,[0,0,0,3],[tf.shape(Y)[0],tf.shape(Y)[1],tf.shape(Y)[2],1]),tf.slice(Y_pred,[0,0,0,3],[tf.shape(Y_pred)[0],tf.shape(Y_pred)[1],tf.shape(Y_pred)[2],1]))

"""
Here I have multiple IOU4 with lamda parameter as the number of pixel in background is larger compare to other class
"""

IOU = IOU1+IOU2+IOU3+lamda*IOU4

optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(-IOU)

n_iterations = 50000

	
train = [0,1,2,3,4,5,6,7,9,10,12,13,14,15,16,17,19]
test = [8,18]

acc_train = []
acc_val = []
acc_test = []

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.36)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess.run(tf.global_variables_initializer())


def Read_data(i):

	"""
	{Data_Loader:images_and_mask_must_be_multiple_of_8}

	Args:
		input:image_position
	
	Returns:	
		output_A(3-D array):Normalized_image
		output_B(3-D array):mask			
	"""

	str1 = ('//home//Drive2//Shanya//Data//0//Random_Init_1//25 1.0//Constructed_Images//'+str(i)+'.npy')
	label = ('Label//'+str(i)+'.npy')		
	label_train = np.load(label)
	data_train = np.load(str1)		
	row = data_train.shape[0]
	column = data_train.shape[1]
	New_row = 8*((row//8))
	New_column = 8*((column//8))
	data_train = data_train[:New_row,:New_column,:]
	data_train = (data_train - data_train.mean())/(data_train.max() - data_train.min())
	label_train = label_train[:New_row,:New_column,:]

	return data_train,label_train


for it_i in range(n_iterations):
	
	if it_i % 50==0:
		print(it_i)
		error = 0
		accuracy_train = 0
		count = 0		
		for itera_train,pos_train in enumerate(train):
			img,label = Read_data(pos_train)
						
			data = []
			label = []
			data.append(img)
			label.append(label)
			accuracy_train+=sess.run(accuracy,feed_dict={X:data, Y: label})
			error+=sess.run(IOU,feed_dict={X:data, Y: label})

			count += 1

		print('acc on data_train',accuracy_train/count)
		acc_trainig.append(accuracy_train/count)
		print('IOU = ',error/count)
				
		accuracy_test = 0
		count = 0		
		for itera_test,pos_test in enumerate(test):
			img,label= Read_data(pos_test)
			data = []
			label = []
			data.append(img)
			label.append(label)
			accuracy_test += sess.run(accuracy,feed_dict={X:data, Y: label})
			count += 1
			
			if accuracy_test/count >= 0.7:		
				ys_pred = np.array(sess.run(Y_pred,feed_dict={X:data_, Y: label_}))
				plt.imshow(ys_pred[0,:,:,:3])
				plt.show()
		print('acc on data_test ',accuracy_test/count)
		acc_test.append(accuracy_test/count)
		np.save('test_accuracy_IOU_with_3_max_pool_BS=3',acc_test)
		np.save('training_accuracy_IOU_with_3_max_pool_BS=3',acc_trainig)
		
	for i in range(n_epoch):
		random.shuffle(train)
		for itera,pos in enumerate(train):
			img,label = Read_data(pos)
			data = []
			label = []
			data.append(img)
			label.append(label)
			data.append(np.flip(img,0))
			label.append(np.flip(label,0))
			data.append(np.flip(img,1))
			label.append(np.flip(label,1))

			sess.run(optimizer,feed_dict={X:data, Y: label})













