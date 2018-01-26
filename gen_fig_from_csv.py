import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


FNAME, SNAME = "dendrogram.csv", "dendrogram_cutting_zzom.png"
TITLE, XAXIS, YAXIS = "Dendrogram Cutting", "Cut Number", "F1-Score"
COL1, COLS2 = 0, [1, 2, 3]
LEGEND = 0

ds = np.genfromtxt(FNAME, delimiter=",")
ds = ds[1:, :]

STEP = ds.shape[0]

fig, ax = plt.subplots()
legends = ["Weighted cluster purity", "Training F1-Score", "Testing F1-Score"]
for COL2 in COLS2:
	print COL1, COL2
	for x in [112]: #range(112, ds.shape[0], STEP):
		xt = range(113) #range(len(ds[x:x+STEP, COL1]))	
		ax.plot(xt, ds[(129-113):, COL2], marker='D')
		ax.set_xticks(np.arange(0, 113, 10))
		#ax.set_yticks(np.arange(0.2, 0.75, 0.05))
		ax.set_xticklabels(np.arange(113, 0, -10))
		#ax.set_xticklabels([ str(x) for x in ds[x:x+STEP, COL1] ], minor=False)
		#legends.append( str(ds[x, LEGEND]) )
ax.legend(legends, loc='best')
if TITLE != "":
	ax.set_title(TITLE)
ax.set_xlabel(XAXIS)
ax.set_ylabel(YAXIS)
plt.savefig(SNAME)
print "Saved figure",SNAME
plt.close()
