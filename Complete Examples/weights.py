import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import distance

from numpy import array, array_equal, allclose

def KNN(X,m,k,ker):
    """
    Function that takes in data X in format
            X = np.array([[x,y]
                        [x,y]])
    where each element inside X is an np.array([]) object itself. We pass in an integer k that 
    determines the number of nearest neighbours, an integer m which is the length of X, and a kernel
    function ker that is a kernel weighting function to decide the weights in our weight matrix
    """
    def arreq_in_list(myarr, list_arrays):
        return next((True for elem in list_arrays if array_equal(elem, myarr)), False)
    k += 1

    dot_products = X @ X.T
    norms = np.diag(dot_products)
    distance_matrix = np.sqrt((norms - 2*dot_products).T + norms)


    nearest_neighbours_idx = [list(i.argsort()[:k]) for i in distance_matrix]

    nearest_neighbours = [list(X[i]) for i in nearest_neighbours_idx]
    # nearest_neighbours = [list(i) for i in nearest_neighbours]

    W = np.reshape([ker(X[i],X[j]) if arreq_in_list(X[i],nearest_neighbours[j]) or arreq_in_list(X[j],nearest_neighbours[i]) else 0 for i in range(len(X)) for j in range(len(nearest_neighbours))],(m,m))
    W -= np.diag(np.diag(W))

    d = np.sum(W,axis=1)
    D = np.diag(d)
    L = D - W
    print("Finished computing graph laplacian")

    return L,W

def proximity(X,m,eps,ker):
    W = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            if distance.euclidean(X[i],X[j]) < eps:
                W[i][j] = ker(X[i],X[j])
            else:
                W[i][j] = 0
    
    d = np.sum(W,axis=1)
    D = np.diag(d)
    D_2 = np.sqrt(D)
    D_inv = np.linalg.inv(D_2)
    L = D - W

    return L,W


