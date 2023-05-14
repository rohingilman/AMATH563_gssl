import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from numpy.random import multivariate_normal as mvn

from scipy.stats import norm
from scipy.optimize import minimize

from math import floor
from math import log

from KNN import KNN

def GSSL_acc(X, y, kvals, k, tau, alpha, lossf):
	# X are data points
	# y are true values
	# kvals are known values for X
	# k is connectedness of graph (num of neighbors)
	# value of tau in computing C
	# value of alpha in computing C
	# lossf is loss function
	# kernel is kernel in weight matrix

	def loss(kvals, y, f):
		if lossf == "probit":
			return -sum([log(norm.cdf(y[j]*f[j])) for j in kvals])
		if lossf == "regression":
			return sum([(y[j]-f[j])**2 for j in kvals])

	m = len(y)

	rbf = lambda x1, x2, gamma: np.exp(-gamma*np.linalg.norm(x1 - x2)**2)

	L, W = KNN(X,k,m,rbf)

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