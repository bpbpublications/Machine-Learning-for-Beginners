import sklearn.datasets as datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import pydotplus
from sklearn.externals.six import StringIO 
from IPython.display import Image
from IPython.display import Image
from IPython.display import SVG
from sklearn.model_selection import train_test_split
  
breast_cancer = datasets.load_breast_cancer()
df = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
target = breast_cancer.target
X_train, X_test, y_train, y_test =train_test_split(df, target, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier(max_depth=3) #max_depth is maximum number of levels in the tree
clf.fit(X_train, y_train)
dotfile = open("D:/dtreeBC.dot", 'w')
dot_data = StringIO()
tree.export_graphviz(clf, 
 out_file=dot_data, 
 class_names=breast_cancer.target_names, # the target names.
 feature_names=breast_cancer.feature_names, # the feature names.
 filled=True, # Whether to fill in the boxes with colours.
 rounded=True, # Whether to round the corners of the boxes.
 special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())