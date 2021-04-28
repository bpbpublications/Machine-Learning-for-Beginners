from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
import numpy as np
import math

#Load Data
IRIS=load_iris()        
X=IRIS.data
y=IRIS.target

#Normalize
max=[]                                                  
min=[]
S=X.shape
for i in range(S[1]):
    	max.append(np.max(X[:,i]))
    	min.append(np.min(X[:,i]))
for i in range(S[1]):
    	for j in range(S[0]):
       	 	X[j,i]=(X[j,i]-min[i])/(max[i]-min[i])

#Prepare test train data 
arr=np.random.permutation(100)
X=IRIS.data[arr,:]
y=np.vstack((np.zeros((50,1)),np.ones((50,1))))
y=y[arr]                                  
X_train=X[:40,:]
X_test=X[40:50,:]
y_train=y[:40]
y_test=y[40:50]
X_train1=X[50:90,:]
X_test1=X[90:100,:]
y_train1=y[50:90]
y_test1=y[90:100]
X_train=np.vstack((X_train,X_train1))
y_train=np.vstack((y_train,y_train1))
X_test=np.vstack((X_test,X_test1))
y_test=np.vstack((y_test,y_test1))

#Classify using SLP
clf=Perceptron(random_state=0)
clf.fit(X_train, y_train)
predicted=clf.predict(X_test)
TP=0
TN=0
FN=0
FP=0
for i in range(len(y_test)):
   	if(y_test[i]==predicted[i]):
        		if(y_test[i]==1):
            			TP+=1
        		else:
            			TN+=1
    	else:
        		if(predicted[i]==1):
            			FP+=1
        		else:
            			FN+=1
acc=(TP+TN)/(TP+TN+FP+FN)      #accuracy
sens=TP/(TP+FN)                #sensitivity
spec=TN/(TN+FP)                #specificity
