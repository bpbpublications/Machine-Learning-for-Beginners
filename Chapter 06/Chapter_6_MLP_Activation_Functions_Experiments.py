from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
import numpy as np
from matplotlib import pyplot as plt

Data=load_breast_cancer()
X=Data.data
y=Data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

y_bar=clf.predict(X_test)

#Identity Activation
a=[]
for i in range(0, 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation= 'identity')
    clf.fit(X_train, y_train)
    y_bar=clf.predict(X_test)
    TP, TN, FP, FN= cal_perf(y_bar,y_test)
    acc1=(TP+TN)/(TP+TN+FP+FN)
    a.append(acc1)
Mean=Average(a)
print(Mean)

#Logistic Activation
a=[]
for i in range(0, 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation= 'logistic')
    clf.fit(X_train, y_train)
    y_bar=clf.predict(X_test)
    TP, TN, FP, FN= cal_perf(y_bar,y_test)
    acc1=(TP+TN)/(TP+TN+FP+FN)
    a.append(acc1)
Mean=Average(a)
print(Mean)


#tanh Activation
a=[]
for i in range(0, 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation= 'tanh')
    clf.fit(X_train, y_train)
    y_bar=clf.predict(X_test)
    TP, TN, FP, FN= cal_perf(y_bar,y_test)
    acc1=(TP+TN)/(TP+TN+FP+FN)
    a.append(acc1)
Mean=Average(a)
print(Mean)

#ReLU Activation
a=[]
for i in range(0, 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation= 'relu')
    clf.fit(X_train, y_train)
    y_bar=clf.predict(X_test)
    TP, TN, FP, FN= cal_perf(y_bar,y_test)
    acc1=(TP+TN)/(TP+TN+FP+FN)
    a.append(acc1)
Mean=Average(a)
print(Mean)


def Average(list): 
    return sum(list)/ len(list) 


def cal_perf(y_pred, y_test):    
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test[i]):
            if(y_pred[i]==1):
                TP+=1
            else:
                TN+=1

        else:
            if(y_pred[i]==1):
                FP+=1
            else:
                FN+=1
    return TP, TN, FP, FN


TP, TN, FP, FN= cal_perf(y_bar,y_test)
acc1=(TP+TN)/(TP+TN+FP+FN)
print(acc1)