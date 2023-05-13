from math import sin
from math import cos

import numpy as np

def spiralPoints(n, t0, t1, nVar = 1.0):
	# n is number of points to be sampled
	# t0 is start of range
	# t1 is end of range
	# nVar is variance of noise, default value 1

	stDv = float(nVar) ** 0.5

	myPoints = []

	for i in range(n):
		t = (t1 - t0)*np.random.rand() + t0 # Get a t value

		x = t*cos(t) # Generate points on the spiral
		y = t*sin(t) # Generate 

		x_n = x + stDv*np.random.randn()
		y_n = y + stDv*np.random.randn()

		myPoints.append((x_n,y_n))

	return np.array(myPoints)