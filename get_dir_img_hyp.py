from loaddata import loaddata,loaddata_spatial, get_train_data
from ssnmf_func import ssnmf, nmf, ssfnnmf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image as pimg
import matplotlib
import os
from spectral import *
from sklearn import svm
import time
from sklearn.semi_supervised import label_propagation

color_val =spy_colors
color_val[0] = color_val[1]
color_val[1] = color_val[2]
color_val[2] = color_val[3]
color_val[3] = [0,0,0]

for inum in range(11, 20):
	data = loaddata(inum, 0.07)
	tot_labels = data[1]
	tot_labels[ tot_labels == 0] = 4
	tot_labels = tot_labels.reshape(data[2], data[3])
	save_rgb("hyp_image_" + str(inum) + ".jpg", tot_labels, colors=color_val)