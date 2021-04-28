from sklearn.linear_model import Perceptron
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
import numpy as np
import math

#Load dataset
dataset=load_breast_cancer()
X=dataset.data
y=dataset.target

#K Fold 
kf=KFold(n_splits=10,random_state=None,shuffle=True)
kf.get_n_splits(X)

#Classification Using Perceptron 
accur=[]
specificity=[]
senstivity=[]
for train_index, test_index in kf.split(X):
    	print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
    	X_train, X_test = X[train_index], X[test_index]
    	y_train, y_test = y[train_index], y[test_index]
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
    		acc=(TP+TN)/(TP+TN+FP+FN)    
    		accur.append(acc)
    		sens=TP/(TP+FN)                
    		senstivity.append(sens)
    		spec=TN/(TN+FP)                
    		specificity.append(spec)
 
#Performance
print(np.mean(accur))
print(np.mean(senstivity))
print(np.mean(specificity))
