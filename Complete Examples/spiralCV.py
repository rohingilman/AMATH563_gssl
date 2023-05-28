import numpy as np

from tqdm import tqdm

from math import pi
from math import dist

from itertools import product

from weights import KNN
from weights import proximity

from accuracy import checkAccuracy
from points import spiralPoints

'''Generates Cluster Data with knownvals'''
def clusters():
	m = 400 # Multiple of 4
	tranges = np.array([[pi,5*pi/2], [3*pi,7*pi/2], [4*pi,9*pi/2]])
	var = 1

	knownvals = [int(j*int(m/8)) for j in range(8)] # Points for which we know the value of the label

	X = spiralPoints(int(m/2), tranges[0,0], tranges[0,1], var)
	X = np.append(X, spiralPoints(int(m/4), tranges[1,0], tranges[1,1], var), axis=0)
	X = np.append(X, spiralPoints(int(m/4), tranges[2,0], tranges[2,1], var), axis=0)

	y = [1 for i in range(int(m/2))]
	y = y + [-1 for i in range(int(m/2))]

	return X, y, knownvals

def numClusters(ews):
	n = 1
	while n < len(ews) and abs(ews[n]) < 1e-14:
		n += 1
	while n < len(ews) and ews[n+1]/ews[n] > 8:
		n += 1
	return n
	


def main():
	'''KNN Graph'''
	# Step 1: Cross Validating the Graph Parameters (for KNN, this is K)
	'''
	# Try with uniform kernel first
	unif = lambda x1, x2: 1

	avgClusts = []
	kvals = [*range(2,10)]
	
	for k in tqdm(kvals):
		sumClusts = 0
		for i in range(10):
			X,y,knownvals = clusters()
			L, W = KNN(X, 400, k, unif)

			ews,evs = np.linalg.eigh(L)

			sumClusts += numClusters(ews)
		avgClusts += [sumClusts/10]
		
	print(avgClusts)
	'''

	# Now let's try with RBF

	params = {
		"k": [4],
		"gamma": [0.3]
		}

	keys, values = zip(*params.items())
	combinations = [dict(zip(keys, p)) for p in product(*values)] # Get all combinations of parameters

	avgClusts = []

	for c in tqdm(combinations):
		k = c["k"]
		gamma = c["gamma"]

		rbf = lambda x1, x2: np.exp(gamma**2*-0.5*dist(x1,x2)**2)

		sumClusts = 0
		for i in tqdm(range(50)):
			X,y,knownvals = clusters()
			L, W = KNN(X, 400, k, rbf)

			ews,evs = np.linalg.eigh(L)

			sumClusts += numClusters(ews)
		avgClusts += [[c,sumClusts/50]]

	print(avgClusts)

	#Step 2: Cross Validating the Model Paramters
	

if __name__=="__main__":
    main()