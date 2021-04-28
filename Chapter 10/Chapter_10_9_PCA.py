from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg

data=load_iris()
Data=data.data
Data1=data.data[50:100,:]
Data2=data.data[100:150,:]
print(Data1.shape)
print(Data2.shape)
index=np.arange(50)
plt.plot(index,Data1[:,0],'rs')
plt.plot(index,Data2[:,0],'bs',)
plt.xlabel('Feature Number')
plt.ylabel('Feature Value')
plt.show()
plt.plot(index,Data1[:,1],'rs')
plt.plot(index,Data2[:,1],'bs',)
plt.xlabel('Feature Number')
plt.ylabel('Feature Value')
plt.show()
plt.plot(index,Data1[:,2],'rs')
plt.plot(index,Data2[:,2],'bs',)
plt.xlabel('Feature Number')
plt.ylabel('Feature Value')
plt.show()
plt.plot(index,Data1[:,3],'rs')
plt.plot(index,Data2[:,3],'bs',)
plt.xlabel('Feature Number')
plt.ylabel('Feature Value')
plt.show()

mean=np.mean(Data, axis=0)
s=np.matmul(np.transpose(Data-mean),(Data-mean))
val, vec=linalg.eig(s)
Data_transformed=np.matmul(Data,vec)
print(Data_transformed.shape)


Data1=Data_transformed[50:100,:]
Data2=Data_transformed[100:150,:]
plt.plot(index,Data1[:,0],'rs')
plt.plot(index,Data2[:,0],'bs',)
plt.xlabel('Feature Number')
plt.ylabel('Feature Value')
plt.show()
plt.plot(index,Data1[:,1],'rs')
plt.plot(index,Data2[:,1],'bs',)
plt.xlabel('Feature Number')
plt.ylabel('Feature Value')
plt.show()
plt.plot(index,Data1[:,2],'rs')
plt.plot(index,Data2[:,2],'bs',)
plt.xlabel('Feature Number')
plt.ylabel('Feature Value')
plt.show()
plt.plot(index,Data1[:,3],'rs')
plt.plot(index,Data2[:,3],'bs',)
plt.xlabel('Feature Number')
plt.ylabel('Feature Value')
plt.show()