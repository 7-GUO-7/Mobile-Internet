import os
folder_name = "D:\\training_negative\\arxiv"
file_list = [name for name in os.listdir(folder_name)]
# print(file_list)

folder_name = "D:\\AAAI_train\\arxiv"
file_list2 = [name for name in os.listdir(folder_name)]
# print(file_list2)

x = []
for i in file_list:
    t = i.split(".")[0] + '.' + i.split(".")[1]
    x.append(t)
y = []
for i in file_list2:
    t = i.split(".")[0] + '.' + i.split(".")[1]
    y.append(t)

for i in y:
    if i not in x:
        print(i)
print(y)


