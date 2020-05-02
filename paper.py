import time
import requests
import random

import os

path = "C:\\Users\\22427\\PycharmProjects\\untitled\\arxiv"
files = os.listdir(path)
savePath = "source"

if not os.path.exists(savePath):
    os.mkdir(savePath)

for filename in files:
    a = filename.split(".")[0]
    b = filename.split(".")[1]
    if b != "jpg":
        selected_paper_id = a + '.' + b + '.pdf'
        print(selected_paper_id)
        r = requests.get('https://arxiv.org/pdf/' + selected_paper_id)
        dpath = os.path.join(savePath, selected_paper_id)
        while r.status_code == 403:
            print(1)
            time.sleep(10 + random.uniform(0, 5))
            r = requests.get('https://arxiv.org/pdf/' + selected_paper_id)
        with open(dpath, "wb") as code:
            code.write(r.content)

