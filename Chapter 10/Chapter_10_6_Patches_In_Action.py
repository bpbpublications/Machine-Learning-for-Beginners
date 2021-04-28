import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def _data(data):
    x=data.data
    y=data.target
    return x,y

def _patch(x, n, mode='avg'):
    s=np.shape(x)
    m=int(s[1]**(1/2))
    x_reshape=[[] for i in range(s[0])]
    for i in range(s[0]):
        x_reshape[i]=np.reshape(x[i], (m,m))
    x_new=np.empty((360, m-(n-1), m-(n-1)))
    s1=np.shape(x_new)
    for i in range(s1[0]):
        for j in range(s1[1]):
            for k in range(s1[2]):
                temp=np.empty((n,n))
                for p in range(n):
                    for q in range(n):
                        temp[p,q]=x_reshape[i][j+p, k+p]
                if(mode=='avg'):
                    x_new[i][j,k]=np.mean(temp)
                elif(mode=='max'):
                    x_new[i][j,k]=np.max(temp)
    return x_new

def _plot(x):
    plt.matshow(x, cmap = 'gray')
    plt.show()


def _acc(y_test, y_predict):
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
    return acc

data=load_digits(n_class=2)
x,y=_data(data)
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.4, random_state=4)
clf=SVC(kernel='linear') #gamma='auto'
clf.fit(x_train, y_train)
y_predict=clf.predict(x_test)
acc=_acc(y_test, y_predict)
print(acc)
