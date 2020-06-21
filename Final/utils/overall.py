import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import os

X = [[] for i in range(5510)]

train_neg = open("10train_neg_pattern2.txt", 'r').readlines()
folder_name = "D:\\training_negative\\arxiv"
file_list = [name for name in os.listdir(folder_name)]
x = []
for i in file_list:
    t = i.split(".")[0] + '.' + i.split(".")[1] + '.pdf'
    x.append([t])

# print(x)
for i in train_neg:
    if len(i) > 0:
        t = i.split(' ')
        if [t[0]] in x:
            n = x.index([t[0]])
            a = "".join(t[1:])
            # print(a)
            d = a.split(']')[0].split('[')[1]
            d2 = a.split(']')[1].split('[')[1]
            d3 = a.split(']')[2].split('[')[1]
            x[n].append(list(eval(d)))
            x[n].append(list(eval(d2)))
            x[n].append(list(eval(d3)))
for i in x:
    if len(i) == 1:
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])

# print(x)
a1 = []
for i in x:
    a1.append(i[1])
# print(a1)

train_neg = open("10train_pos_pattern2.txt", 'r').readlines()
folder_name = "D:\\AAAI_train_conference_pdf"
file_list = [name for name in os.listdir(folder_name)]
x = []
for i in file_list:
    t = i.split(".")[0] + '.' + i.split(".")[1]
    x.append([t])

for i in train_neg:
    if len(i) > 0:
        t = i.split(' ')
        if [t[0]] in x:
            n = x.index([t[0]])
            a = "".join(t[1:])
            # print(a)
            d = a.split(']')[0].split('[')[1]
            d2 = a.split(']')[1].split('[')[1]
            d3 = a.split(']')[2].split('[')[1]
            x[n].append(list(eval(d)))
            x[n].append(list(eval(d2)))
            x[n].append(list(eval(d3)))
for i in x:
    if len(i) == 1:
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])
# print(x)
for i in x:
    # print(i)
    a1.append(i[1])

print(a1)
# t1 = []
# for it in a1:
#     t1.append([it[0]])
# print(t1)
y = [0 for i in range(2650)] + [1 for j in range(2860)]

classifier = joblib.load('figure_0.80.model')
predicted = classifier.predict(a1)
classifier.fit(a1, y)
print(predicted)
print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y, predicted)))
for i, j in enumerate(predicted):
    X[i].append(j)
print(X)

train_neg = open("10train_neg_pattern2.txt", 'r').readlines()
folder_name = "D:\\training_negative\\arxiv"
file_list = [name for name in os.listdir(folder_name)]
x = []
for i in file_list:
    t = i.split(".")[0] + '.' + i.split(".")[1] + '.pdf'
    x.append([t])

# print(x)
for i in train_neg:
    if len(i) > 0:
        t = i.split(' ')
        if [t[0]] in x:
            n = x.index([t[0]])
            a = "".join(t[1:])
            # print(a)
            d = a.split(']')[0].split('[')[1]
            d2 = a.split(']')[1].split('[')[1]
            d3 = a.split(']')[2].split('[')[1]
            x[n].append(list(eval(d)))
            x[n].append(list(eval(d2)))
            x[n].append(list(eval(d3)))
for i in x:
    if len(i) == 1:
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])

# print(x)
a1 = []
for i in x:
    a1.append(i[2])
# print(a1)

train_neg = open("10train_pos_pattern2.txt", 'r').readlines()
folder_name = "D:\\AAAI_train_conference_pdf"
file_list = [name for name in os.listdir(folder_name)]
x = []
for i in file_list:
    t = i.split(".")[0] + '.' + i.split(".")[1]
    x.append([t])

for i in train_neg:
    if len(i) > 0:
        t = i.split(' ')
        if [t[0]] in x:
            n = x.index([t[0]])
            a = "".join(t[1:])
            # print(a)
            d = a.split(']')[0].split('[')[1]
            d2 = a.split(']')[1].split('[')[1]
            d3 = a.split(']')[2].split('[')[1]
            x[n].append(list(eval(d)))
            x[n].append(list(eval(d2)))
            x[n].append(list(eval(d3)))
for i in x:
    if len(i) == 1:
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])
# print(x)
for i in x:
    # print(i)
    a1.append(i[2])

# print(a1)
# t1 = []
# for it in a1:
#     t1.append([it[0]])
# print(t1)
y = [0 for i in range(2650)] + [1 for j in range(2860)]

classifier = joblib.load('table_0.74.model')
predicted = classifier.predict(a1)
classifier.fit(a1, y)
# print(predicted)
print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y, predicted)))
for i, j in enumerate(predicted):
    X[i].append(j)
# print(X)

train_neg = open("10train_neg_pattern2.txt", 'r').readlines()
folder_name = "D:\\training_negative\\arxiv"
file_list = [name for name in os.listdir(folder_name)]
x = []
for i in file_list:
    t = i.split(".")[0] + '.' + i.split(".")[1] + '.pdf'
    x.append([t])

# print(x)
for i in train_neg:
    if len(i) > 0:
        t = i.split(' ')
        if [t[0]] in x:
            n = x.index([t[0]])
            a = "".join(t[1:])
            # print(a)
            d = a.split(']')[0].split('[')[1]
            d2 = a.split(']')[1].split('[')[1]
            d3 = a.split(']')[2].split('[')[1]
            x[n].append(list(eval(d)))
            x[n].append(list(eval(d2)))
            x[n].append(list(eval(d3)))
for i in x:
    if len(i) == 1:
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])

# print(x)
a1 = []
for i in x:
    a1.append(i[3])
# print(a1)

train_neg = open("10train_pos_pattern2.txt", 'r').readlines()
folder_name = "D:\\AAAI_train_conference_pdf"
file_list = [name for name in os.listdir(folder_name)]
x = []
for i in file_list:
    t = i.split(".")[0] + '.' + i.split(".")[1]
    x.append([t])

for i in train_neg:
    if len(i) > 0:
        t = i.split(' ')
        if [t[0]] in x:
            n = x.index([t[0]])
            a = "".join(t[1:])
            # print(a)
            d = a.split(']')[0].split('[')[1]
            d2 = a.split(']')[1].split('[')[1]
            d3 = a.split(']')[2].split('[')[1]
            x[n].append(list(eval(d)))
            x[n].append(list(eval(d2)))
            x[n].append(list(eval(d3)))
for i in x:
    if len(i) == 1:
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])
        i.append([0 for i in range(10)])
# print(x)
for i in x:
    # print(i)
    a1.append(i[3])

# print(a1)
# t1 = []
# for it in a1:
#     t1.append([it[0]])
# print(t1)
y = [0 for i in range(2650)] + [1 for j in range(2860)]

classifier = joblib.load('formula_0.83.model')
predicted = classifier.predict(a1)
classifier.fit(a1, y)
# print(predicted)
print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y, predicted)))
for i, j in enumerate(predicted):
    X[i].append(j)
print(X)


train_neg = open("train_neg_pattern2.txt", 'r').readlines()
train_pos = open("train_pos_pattern2.txt", "r").readlines()
a = []


def take(elem):
    return elem[0]


for i in train_neg:
    if len(i) > 0:
        t = i.split(' ')
        x1 = int(t[4])
        a.append([t[0], x1])
a.sort(key=take)

c = []
for it in a:
    c.append([it[1]])

d = []
for i in train_pos:
    if len(i) > 0:
        t = i.split(' ')
        x1 = int(t[4])
        d.append([t[0], x1])
d.sort(key=take)
for it in d:
    c.append([it[1]])

y = [0 for i in range(2650)] + [1 for j in range(2860)]

classifier = joblib.load('reference_svm_0.81.model')
predicted = classifier.predict(c)
classifier.fit(c, y)
# print(predicted)
print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y, predicted)))
for i, j in enumerate(predicted):
    X[i].append(j)
print(X)

train_neg = open("negative.txt", 'r').readlines()
train_pos = open("positive.txt", "r").readlines()

b = []

for i in train_neg:
    if len(i) > 0:
        t = i.split(',')
        x1 = []
        if len(t) == 1:
            x1.append(0)
            x1.append(0)
            x1.append(0)
            x1.append(0)
            b.append(x1)
            continue
        x1.append(float(t[0]))
        if len(t) == 2:
            x1.append(0)
            x1.append(0)
            x1.append(0)
        if len(t) == 3:
            x1.append(float(t[1]))
            x1.append(0)
            x1.append(0)
        if len(t) == 4:
            x1.append(float(t[1]))
            x1.append(float(t[2]))
            x1.append(0)
        if len(t) > 4:
            x1.append(float(t[1]))
            x1.append(float(t[2]))
            x1.append(float(t[3]))
        b.append(x1)
        # print(x1)
# print(len(x))

for i in train_pos:
    if len(i) > 0:
        t = i.split(',')
        x1 = []
        if len(t) == 1:
            x1.append(0)
            x1.append(0)
            x1.append(0)
            x1.append(0)
            b.append(x1)
            continue
        x1.append(float(t[0]))
        if len(t) == 2:
            x1.append(0)
            x1.append(0)
            x1.append(0)
        if len(t) == 3:
            x1.append(float(t[1]))
            x1.append(0)
            x1.append(0)
        if len(t) == 4:
            x1.append(float(t[1]))
            x1.append(float(t[2]))
            x1.append(0)
        if len(t) > 4:
            x1.append(float(t[1]))
            x1.append(float(t[2]))
            x1.append(float(t[3]))
        b.append(x1)

print(len(b))
y = [0 for i in range(2650)] + [1 for j in range(2860)]

classifier = joblib.load('frequency_0.68.model')
predicted = classifier.predict(b)
classifier.fit(b, y)
# print(predicted)
print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y, predicted)))

for i, j in enumerate(predicted):
    X[i].append(j)
print(X)

for d in range(3):
    classifier = MLPClassifier(solver='adam', hidden_layer_sizes=(4,), alpha=0.1, max_iter=600)
    # classifier = svm.SVC(kernel='rbf', class_weight='balanced')
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(y_test, predicted)))
    # joblib.dump(classifier, 'mixture-five' + str(d) + '.model')

folder_name = "result"
file_list = [name for name in os.listdir(folder_name)]
file_list = [folder_name + '/' + str(name) for name in file_list]
x = []

for i, file in enumerate(file_list):
    st = open(file, 'r').read()
    X[i].append(eval(st)/100)


train = open("lstm.txt", 'r').readlines()
p = []
for i in train:
    if len(i) > 0:
        i = i.rstrip('\n')
        i = i.rstrip(']')
        while i[-1] == ' ':
            i = i.rstrip(' ')
        p.append(float(i.split(' ')[-1]))
# print(p)

for i, j in enumerate(p):
    X[i].append(j)
print(X[-120])

for d in range(3):
    classifier = MLPClassifier(solver='adam', hidden_layer_sizes=(4,), alpha=0.1, max_iter=600)
    # classifier = svm.SVC(kernel='rbf', class_weight='balanced')
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(y_test, predicted)))
    joblib.dump(classifier, 'mixture-resnet' + str(d) + '.model')
