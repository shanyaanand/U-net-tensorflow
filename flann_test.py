from pyflann import *
from numpy import *
from numpy.random import *
import time

start_time = time.time()


dataset = rand(300000, 4)
testset = rand(1000, 4)
flann = FLANN()
result,dists = flann.nn(dataset,testset,5,algorithm="kmeans",
branching=32, iterations=7, checks=16);

print("--- %s seconds ---" % (time.time() - start_time))
