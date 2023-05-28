import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize

from math import log

from weights import KNN
from weights import proximity

def KNN_acc(X, y, kvals, k, tau, alpha, lossf, kernel):
	"""	
	X are data points
	y are true values
	kvals are known values for X
	k is connectedness of graph (num of neighbors)
	value of tau in computing C
	value of alpha in computing C
	lossf is loss function
	kernel is kernel in weight matrix
	"""

	def loss(kvals, y, f):
		if lossf == "probit":
			return -sum([log(norm.cdf(y[j]*f[j])) for j in kvals])
		if lossf == "regression":
			return sum([(y[j]-f[j])**2 for j in kvals])

	m = len(y)
	print("entering in KNN function")
	L, W = KNN(X, m, k, kernel)

	lamb = (tau**(2*alpha))/2
	C = np.linalg.matrix_power(((L + (tau**2)*np.eye(m))),-alpha)
	C_inv = np.linalg.inv(C)

	def regular(lamb,C_inv,f):
		f_T = np.array(f).T
		return lamb*f_T.dot(C_inv).dot(f)

	def to_minimize(f,kvals,y,lamb,C_inv):
		return loss(kvals,y,f) + regular(lamb,C_inv,f)

	f0 = np.zeros(m)

	print("Starting the minimization")
	result = minimize(to_minimize, f0, args=(kvals,y,lamb,C_inv), method='BFGS')

	print("After minimizing")
	f_star = result.x

	y_pred = np.sign(f_star)

	accuracy = sum([x[0] == x[1] for x in zip(y_pred,y)])/m

	return accuracy

def Prox_acc(X, y, kvals, eps, tau, alpha, lossf, kernel):
	"""	
	X are data points
	y are true values
	kvals are known values for X
	k is connectedness of graph (num of neighbors)
	value of tau in computing C
	value of alpha in computing C
	lossf is loss function
	kernel is kernel in weight matrix
	"""

	def loss(kvals, y, f):
		if lossf == "probit":
			return -sum([log(norm.cdf(y[j]*f[j])) for j in kvals])
		if lossf == "regression":
			return sum([(y[j]-f[j])**2 for j in kvals])

	m = len(y)

	L, W = proximity(X,m,eps, kernel)

	lamb = (tau**(2*alpha))/2
	C = np.linalg.matrix_power(((L + (tau**2)*np.eye(m))),-alpha)
	C_inv = np.linalg.inv(C)

	def regular(lamb,C_inv,f):
		f_T = np.array(f).T
		return lamb*f_T.dot(C_inv).dot(f)

	def to_minimize(f,kvals,y,lamb,C_inv):
		return loss(kvals,y,f) + regular(lamb,C_inv,f)

	f0 = np.zeros(m)

	result = minimize(to_minimize, f0, args=(kvals,y,lamb,C_inv), method='BFGS')

	f_star = result.x

	y_pred = np.sign(f_star)

	accuracy = sum([x[0] == x[1] for x in zip(y_pred,y)])/m

	return accuracy