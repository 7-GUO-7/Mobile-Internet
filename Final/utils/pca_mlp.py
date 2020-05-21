import os
import cv2
import numpy as np
from numpy.core.umath_tests import inner1d
from sklearn.decomposition import PCA
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


folder_name = "pic"
file_list = [name for name in os.listdir(folder_name)]
file_list = [folder_name + '/' + str(name) for name in file_list]


x = []

for file in file_list:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    x.append(img.flatten(1))

for d1 in [1e-4, 1e-2, 0.5]:
    for d2 in [4]:
        X_p = PCA(n_components=14, svd_solver='randomized',
                  whiten=True).fit_transform(np.array(x))
        # pca.fit(x)
        # print(pca.explained_variance_ratio_)
        # X_p = pca.transform(x)
        # print(X_p)
        y = [0 for i in range(3198)] + [1 for j in range(2860)]

        # f1-0.68
        classifier = MLPClassifier(solver='adam', hidden_layer_sizes=(36, 8, 4), alpha=0.5, max_iter=800)

        X_train, X_test, y_train, y_test = train_test_split(
            X_p, y, test_size=0.2, shuffle=True)
        classifier.fit(X_train, y_train)

        predicted = classifier.predict(X_test)

        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(y_test, predicted)))
