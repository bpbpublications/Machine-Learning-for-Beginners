import numpy as np
from sklearn import datasets
data= datasets.load_iris()


def load_data():
    Data=load_breast_cancer()
    X=Data.data
    y=Data.target
    return (X, y)    

def cal_acc(y_test, y_predict):
    tp=0 
    tn=0
    fp=0
    fn=0
    s=np.shape(y_test)
    for i in range (s[0]):
        o1=y_predict[i]
        y1=y_test[i]
        if(o1==1 and y1==1):
            tp+=1
        elif(o1==0 and y1==0):
            tn+=1
        elif(o1==1 and y1==0):
            fp+=1
        else:
            fn+=1
    acc=(tp+tn)/(tp+tn+fp+fn)*100
    return(acc)

X, y=load_data()
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=4)
clf=SVC(kernel='linear') #gamma='auto' change parameters and observe results
clf.fit(X_train, y_train)
y_predict=clf.predict(X_test)
accuracy=cal_acc(y_test, y_predict)
print(accuracy)


kf=KFold(n_splits=10)
kf
kf.get_n_splits(X)
acc=[]
for train_i,test_i in kf.split(X):
    X_train,X_test=X[train_i],X[test_i]
    y_train,y_test=y[train_i],y[test_i]
    clf=SVC(kernel='linear') #gamma='auto'
    clf.fit(X_train, y_train)
    y_predict=clf.predict(X_test)
    accuracy=cal_acc(y_test, y_predict)
    acc.append(accuracy)
print(np.mean(acc))


plt.plot(acc)
plt.show()
print(acc)
