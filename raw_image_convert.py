from loaddata import loaddata, get_train_data
from ssnmf_func import ssnmf
import scipy.io as sci
import numpy as np
import time

##Sample values that you can pass to the get_nmf_cube function
rank, bg_param = 10, 0.07
start_time = time.time()

def get_raw_image(img):
	bg_param = 0.07
	start_time = time.time()
	data_pi = loaddata(img, bg_param)
	label_mat = np.reshape(data_pi[1], (data_pi[2], data_pi[3]))
	raw_image = np.zeros((data_pi[2], data_pi[3], data_pi[0].shape[0]))
	for i in range(data_pi[0].shape[1]):
		r_idx, c_idx = (i/data_pi[3]), (i%data_pi[3])
		raw_image[r_idx, c_idx, :] = data_pi[0][:, i]
	# print data_pi[0].shape, raw_image.shape
	# print label_mat.shape
	print img,time.time() - start_time,"seconds"
	return raw_image, label_mat

for i in range(1, 10):
	a, b = get_raw_image(i)
	sci.savemat('raw_normal_data_' + str(i) + '.mat', dict(normal_tensor=a, normal_label=b))
	a, b = get_raw_image(i+10)
	sci.savemat('raw_hyperplasia_data_' + str(i) + '.mat', dict(hyperplasia_tensor=a, hyperplasia_label=b))