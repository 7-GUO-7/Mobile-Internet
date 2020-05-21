import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from PIL import Image

# folder_name = "training"
# savePath = "pic"
#
# if not os.path.exists(savePath):
#     os.mkdir(savePath)
# file_list = [name for name in os.listdir(folder_name)]
# file_list = [folder_name + '/' + str(name) for name in file_list]
#
# i = 0
# for file in file_list:
#     print(file)
#     i += 1
#     img = Image.open(file)
#     out = img.resize((200, 200),Image.ANTIALIAS)
#     path = os.path.join(savePath, str(i) + ".jpg")
#     out.save(path)

folder_name = "pic"
file_list = [name for name in os.listdir(folder_name)]
file_list = [folder_name + '/' + str(name) for name in file_list]


x = []

for file in file_list:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    x.append(img.flatten(1))

# n_components = 20
for n in [12]:
    X_p = PCA(n_components=n, svd_solver='randomized',
              whiten=True).fit_transform(np.array(x))
    # pca.fit(x)
    # print(pca.explained_variance_ratio_)
    # X_p = pca.transform(x)
    # print(X_p)
    y = [0 for i in range(3198)] + [1 for j in range(2860)]

    # f1-0.65
    classifier = svm.SVC(kernel='rbf', class_weight='balanced')

    # f1-0.61
    # classifier = svm.SVC(kernel='linear', class_weight='balanced')

    X_train, X_test, y_train, y_test = train_test_split(
        X_p, y, test_size=0.2, shuffle=True)
    classifier.fit(X_train, y_train)

    predicted = classifier.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, predicted)))



