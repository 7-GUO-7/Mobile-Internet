import matplotlib.pyplot as plt
import os

# train_neg = open("train_neg_pattern2.txt", 'r').readlines()
# train_pos = open("train_pos_4d.txt", "r").readlines()
#
# x1 = []
# x2 = []
#
# for i in train_neg:
#     if len(i) > 0:
#         t = i.split(' ')
#         if int(t[4]) == 0:
#             continue
#         x1.append(int(t[4]))
#
# for i in train_pos:
#     if len(i) > 0:
#         t = i.split(' ')
#         if int(t[3]) == 0:
#             continue
#         x2.append(int(t[3]))
#
# labels = ['neg', 'pos']
# bins = [i for i in range(0, 30)]
# print(len(x1), len(x2))
# x1 += [0 for i in range(220)]
# x2 += [0 for i in range(20)]
# print(x1)
# plt.hist([x1, x2], bins=bins, label=labels, color=['r', 'b'])
#
# plt.show()

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
print(a1)

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
print(len(a1))




