import numpy as np
from loaddata import loaddata

if __name__ == "__main__":
	for i in [10]:#range(1, 10):
		data, _, _, _ = loaddata(i)
		print "Processing Normal",str(i)
		with open("normal_psd.csv", "a+") as f:
			f.write( ",".join( [str(x) for x in np.linalg.norm(data, axis=0) ] ) + "\n"  )
		data, _, _, _ = loaddata(i+10)
		print "Processing Hyperplasia", str(i)
		with open("hyp_psd.csv", "a+") as f:
			f.write( ",".join( [str(x) for x in np.linalg.norm(data, axis=0) ] ) + "\n"  )

