from sklearn.datasets import load_iris
import numpy as np
from sklearn.utils import shuffle

#Loading the data
Data=load_iris()
X=Data.data
Y=Data.target
x=X[:100,:]
y=Y[:100]
x, y = shuffle(x, y, random_state=0)
x_train=x[:80,:]
y_train=y[:80]
x_test=x[80:,:]
y_test=y[80:]
alpha=0.01
s=X.shape

#Initial weights and bais
w=np.random.rand(1,s[1])
b=np.random.rand()
n=len(x_train)

#Learning 
for i in range(n):
   	 x1=x_train[i,:]
    	u=np.matmul(X1,np.transpose(w))
    	v=1/(1+np.exp(-1*u))
    	if (v>0.99):
        		o1=1
    	else:
        		o1=0
    	print(v,' ',y_train[i])
    	y1=y_train[i]
    	w=w-alpha*(o1-y1)*x1
    	b=b-alpha*(o1-y1)
tp=0 
tn=0
fp=0
fn=0
for i in range(20):
    		x1=x_test[i,:]
    		u=np.matmul(x1,np.transpose(w))+b
    		v=1/(1+np.exp(-1*u))
    		if (v>0.5):
        			o1=1
    		else:
        			o1=0
    		y1=y_test[i]
    		if(o1==1 & y1==1):
        			tp+=1
    		elif(o1==0 & y1==0):
        			tn+=1
    		elif(o1==1 & y1==0):
        			fp+=1
   		 else:
        			fn+=1
        
acc=(tp+tn)/(tp+tn+fp+fn)*100
