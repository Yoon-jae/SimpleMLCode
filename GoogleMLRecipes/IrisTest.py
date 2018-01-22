from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

import numpy as np



## Import dataset.
iris = load_iris()


# print iris.feature_names
# print iris.target_names  # Label
# print iris.data[0]
# print iris.target[0]


# for i in range(len(iris.target)):
#     print "Example %d: lable %s, features %s" % (i, iris.target[i], iris.data[i])


## Train a classifier

# training data, Delete datas which index is 0, 50, 100 for test data

test_idx = [0, 50, 100]
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))


## Predice label for new flower.

# viz code

dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")

