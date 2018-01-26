import numpy as np
from spectral import *
import spectral.io.envi as envi

for i in range(1, 21):
	print "Processing",i
	if i < 11:
		str1 = ("Normal%d.hdr" % i)
		str2 = ("Normal%d" % i)
	else:
		i1 = (i%11) + 1
		str1 = ("Hyperplasia%d.hdr" % i1)
		str2 = ("Hyperplasia%d" % i1)
	img = envi.open(str1, str2).load()
	img = np.asarray(img)
	with open("img_sizes.csv", "a+") as f:
		if i < 11:
			f.write("Normal" + str(i) + "," + str(img.shape[0]) + "," + str(img.shape[1]) + "\n" )
		else :
			f.write("Hyperplasia" + str(i1) + "," + str(img.shape[0]) + "," + str(img.shape[1]) + "\n" )
	
