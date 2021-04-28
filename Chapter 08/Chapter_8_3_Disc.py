import numpy as np
from sklearn import datasets
data= datasets.load_iris()



X=data.data[:100,:]
X=np.array(X)
print(X.shape)
y=data.target[:100]
y=np.array(y)
print(y.shape)

n=int(input('Enter the value of n \t:'))


for i in range (X.shape[1]):
    x1=X[:,i]
    max1=np.max(x1)
    min1=np.min(x1)
    step=(max1-min1)/n
    print(max1, ' ',min1, ' ',step)
    for k in range (n):
        a=min1+(step*k)
        b=min1+(step*(k+1))
        for j in range(X.shape[0]):
            if ((X[j,i]>=a) and (X[j,i]<=b)):
                X[j,i]=k
print(X)   
    