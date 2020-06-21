import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

data = open("neg3.txt", 'r').read()
line = data.split('\n')
x = []
for i in line:
    if len(i) > 0:
        x.append(eval(i))

data2 = open("pos3.txt", 'r').read()
line = data2.split('\n')
for i in line:
    if len(i) > 0:
        x.append(eval(i))
print(len(x))
y = [0 for i in range(2634)] + [1 for i in range(2860)]

for d in range(10):
    for m in [2]:
        classifier = MLPClassifier(solver='adam', hidden_layer_sizes=(4, 2), alpha=0.1, max_iter=600)
        # classifier = svm.SVC(kernel='rbf', class_weight='balanced')
        X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, shuffle=True)
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)

        print("Classification report for classifier %s:\n%s\n"
                % (classifier, metrics.classification_report(y_test, predicted)))

        joblib.dump(classifier, 'figure' + str(d) + '.model')
