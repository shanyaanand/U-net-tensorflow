import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from PIL import Image as pimg
import numpy as np
from loaddata import loaddata

if __name__ == "__main__":
	for i in range(1, 10):
		fig = plt.figure()
		fig.suptitle( ("Foreground heatmap for Normal %d" % i) )
		for _ix, x in enumerate([0.001, 0.01, 0.05, 0.07, 0.15, 0.35]):#enumerate([0.001, 0.15, 0.35, 0.5, 0.7, 0.9]):
			data, y, s0, s1 = loaddata(i, x)
			print "Processing Normal",str(i)
			
			norm_data = np.linalg.norm(data, axis=0)
			norm_data[ y == 4 ] = 0
			norm_data = np.uint8( 255 * np.reshape(norm_data, (s0, s1)) )

			ax= plt.subplot( ("23%d" % (1 + _ix) ) )
			ax.set_title ( ("%.3f" % x) )
			ax.axis('off')
			ax.imshow(norm_data) #, cmap="hot")
		plt.savefig( ("PAPER_bg_threshold_masks/heatmap-normal-%d.png" % i), bbox_inches='tight' )
		plt.close()
		
		
		fig = plt.figure()
		fig.suptitle( ("Foreground heatmap for Hyperplasia %d" % i) )
		for _ix, x in enumerate([0.001, 0.01, 0.05, 0.07, 0.15, 0.35]):#enumerate([0.001, 0.15, 0.35, 0.5, 0.7, 0.9]):
			data, y, s0, s1 = loaddata(i+10, x)
			print "Processing Hyperplasia",str(i)
			
			norm_data = np.linalg.norm(data, axis=0)
			norm_data[ y == 4 ] = 0
			norm_data = np.uint8( 255 * np.reshape(norm_data, (s0, s1)) )

			ax= plt.subplot( ("23%d" % (1 + _ix) ) )
			ax.set_title ( ("%.3f" % x) )
			ax.axis('off')
			ax.imshow(norm_data) #, cmap="hot")
		plt.savefig( ("PAPER_bg_threshold_masks/heatmap-normal-%d.png" % i), bbox_inches='tight' )
		plt.close()

