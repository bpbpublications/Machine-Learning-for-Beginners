from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import scipy.linalg

Data=load_iris()
X=Data.data
temp1=X[:3,:]
temp2=X[50:53, :]
X=np.vstack((temp1, temp2))
print(X)

dist=euclidean_distances(X, X)
print(dist)

s=np.sum(dist, axis=1)
D=np.zeros((X.shape[0],X.shape[0]))
for i in range(D.shape[0]):
    D[i,i]=s[i]
print(D)

L=dist-D

[v1, v2]=scipy.linalg.eig(L)
print(v1)
print(v2)
#Look for second smallest 
cut=v2[:,1]

cut