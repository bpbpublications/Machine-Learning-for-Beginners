from sklearn.linear_model import Perceptron
from sklearn.datasets import load_breast_cancer
import numpy as np
import math

#Load Data
dataset=load_breast_cancer()       #constructor called
X=dataset.data
y=dataset.target
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

#Train test data
X_train=X[:400,:]
X_test=X[400:,:]
y_train=y[:400]
y_test=y[400:]

#Classify using Perceptron 
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
