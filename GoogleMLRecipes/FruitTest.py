from sklearn import tree

# features = [[140,"smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
# labels = ["apple", "apple", "orange", "orange"]
features = [[140,1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

## Train Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels) # Find patterns in data

print (clf.predict([[150,0]]))

