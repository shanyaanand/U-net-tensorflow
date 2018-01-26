import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from loaddata import loaddata
import numpy as np

if __name__ == "__main__":
	for img in range(1, 10):
		fig = plt.figure()
		fig.suptitle( ("Background masks for Normal %d" % img) )
		for _ix, x in enumerate([0.001, 0.15, 0.35, 0.5, 0.7, 0.9]):#enumerate([0.001, 0.01, 0.05, 0.07, 0.15, 0.35]):
			_, y, s0, s1 = loaddata(img, x)
			w = np.array( np.reshape( y != 4, (s0, s1)), dtype=int )

			ax = plt.subplot( ("23%d" % (1 + _ix) ) )
			ax.set_title ( ("%.3f" % x) )
			ax.axis('off')
			ax.imshow(w)
		plt.savefig( ("PAPER_bg_threshold_masks_spectral/threshold-normal-%d.png" % img), bbox_inches='tight' )
		plt.close()
		
		fig = plt.figure()
		fig.suptitle( ("Background masks for Hyperplasia %d" % img) )
		for _ix, x in enumerate([0.001, 0.15, 0.35, 0.5, 0.7, 0.9]): #enumerate([0.001, 0.01, 0.05, 0.07, 0.15, 0.35]):
			_, y, s0, s1 = loaddata(img+10, x)
			w = np.array( np.reshape( y != 4, (s0, s1)), dtype=int )

			ax = plt.subplot( ("23%d" % (1 + _ix) ) )
			ax.set_title ( ("%.3f" % x) )
			ax.axis('off')
			ax.imshow(w)
		plt.savefig( ("PAPER_bg_threshold_masks_spectral/threshold-hyp-%d.png" % img), bbox_inches='tight' )
		plt.close()
