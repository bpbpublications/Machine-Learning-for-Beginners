from sklearn.datasets import load_iris
from sklearn import tree
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#from sklearn.tree.export import export_text
Data = load_breast_cancer()
X=Data.data
y=Data.target
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3)
print(X_train.shape, X_test.shape,y_train.shape, y_test.shape)
clf = DecisionTreeClassifier(random_state=0, max_depth=2)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
(TP, TN, FP, FN)=eval(y_pred,y_test)



print(TP, TN, FP, FN)


