import os
from tika import parser


savePath = "txt"

if not os.path.exists(savePath):
    os.mkdir(savePath)


folder_name = "arxiv"
file_list = [name for name in os.listdir(folder_name)]
file_list = [folder_name + '/' + str(name) for name in file_list]
i = 1
for file in file_list:
    raw = parser.from_file(file)
    text = raw['content']
    text = str(text)
    l1 = text.split('\n')
    l2 = []
    for line in l1:
        if len(line) > 10:
            l2.append(line)
    dpath = os.path.join(savePath, str(i) + ".txt")
    with open(dpath, "w", encoding='utf-8') as code:
        for line in l2:
            code.write(str(line))
            code.write('\n')
    i += 1
