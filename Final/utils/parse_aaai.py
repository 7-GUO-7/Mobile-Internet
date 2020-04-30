import urllib.request
import http.cookiejar
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import time
import requests
import random
import os

source = 'AAAI_train/conference'
source_list = os.listdir(source)

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gb2313,utf-8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36",
    "Connection": "keep-alive",
    'referer':'http://www.aaai.org/'
}
#设置cookie
currPath = os.path.dirname(os.path.realpath('__file__'))
savePath = os.path.join(currPath, 'data')

if not os.path.exists(savePath):
    os.mkdir(savePath)

cjar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cjar))
headall = []
# 将制定格式的headers信息添加好
for key,value in headers.items():
    item = (key,value)
    headall.append(item)
opener.addheaders = headall
# 将opener安装为全局
urllib.request.install_opener(opener)

print(len(source_list))
for img in source_list:

    _file, ext = os.path.splitext(img)
    myfile = _file.split('-')
    year, pid = myfile[0][4:], myfile[1] #get aaai year and paper id
    dstpath = os.path.join(savePath, _file+'.pdf')
    try:
        url = "https://www.aaai.org/ocs/index.php/AAAI/AAAI"+year+"/paper/viewPaper/"+pid+"/"
        download = "https://www.aaai.org/ocs/index.php/AAAI/AAAI"+year+"/paper/download/"+pid+"/"

        # dstpath = os.path.join(savePath, _file+".pdf")
        data = urllib.request.urlopen(url).read()
        html = BeautifulSoup(data,'html.parser')
        pdf = html.find("div", {"id":"paper"})
        PDFurl = pdf.a.attrs['href']

        pdfID,paperID = str(PDFurl).split('/')[-1], str(PDFurl).split('/')[-2]
        downloadURL = download + pdfID
        r = requests.get(downloadURL)
        while r.status_code == 403:
            print(1)
            time.sleep(10 + random.uniform(0, 10))
            r = requests.get(downloadURL)
        with open(dstpath, "wb") as code:
            code.write(r.content)
    except Exception as e:
        print(e, _file)
        pass
# print(PDFurl)