from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import string
import PyPDF2
import pdfplumber
import os
import re
import sys
from tika import parser

pdf = "aaai18-17305.pdf"

score_d = {
    '=': 1.2,
    '(': 0.2,
    ')': 0.3,
    '[': 0.3,
    ']': 0.3,
    '−': 0.6,
    '+': 1.5,
    '-': 0.6,
    '/': 1.5,
    '^': 1.1,
    '|': 0.9,
    '': 0
}

formula_pattern = re.compile(r'=|/|−|-|^|\||\[|\]|\+|\(\d{1,2}\)|\(\w\)|\(')
exclude_pattern = re.compile(r'\+\+|http|--|==|pdf')

def cal_score(r):
    s = 0
    for t in r:
        if len(t) > 1:
            s += 1
        else:
            s += score_d[t]
    return s

def get_formula_num(text):
    cnt = 0
    l = []
    tmp = ''
    for t in text.split('\n'):
        if len(t) <= 5:
            tmp += t
        else:
            if tmp != '':
                l.append(tmp)
                tmp = ''
            l.append(t)
    for t in l:
        p = exclude_pattern.findall(t)
        if len(t) > 100 or len(p) > 0: continue
        res = formula_pattern.findall(t)
        s = cal_score(res)
        if s > 3:
            cnt += 1
    return cnt

def getHotWords(title):
    word_dict = {}
    for w in title:
        w = w.lower()
        w = WordNetLemmatizer().lemmatize(w)
        if not re.search(r'[0-9]|^\w$', w):
            if w not in stopWords and w not in string.punctuation and wordnet.synsets(w): 
                if w not in word_dict.keys():
                    word_dict[w] = 1
                else:
                    word_dict[w] += 1
    return word_dict

def numToFrequency(wdcnt, hotwords):
    res = sorted(hotwords.items(), key=lambda item:item[1], reverse=True)
    for key in hotwords.keys():
        hotwords[key] /= wdcnt
    return hotwords

def countHotWords(hotwords, text):
    wdnum = 0
    for line in text:
        if(line == "References"):
            break
        words = word_tokenize(line)
        wdnum += len(words)
        for w in words:
            w = w.lower()
            w = WordNetLemmatizer().lemmatize(w)
            if not re.search(r'[0-9]|^\w$', w):
                if w not in stopwords.words('english') and w not in string.punctuation and wordnet.synsets(w) and w in words:
                    if w in hotwords.keys():
                        hotwords[w] += 1
    return wdnum, hotwords

def getWordsf(title, text):
    hotwords = getHotWords(title)
    wdnum, hotwords = countHotWords(hotwords, text[1:])
    wordf = numToFrequency(wdnum, hotwords)
    return wordf

def countReferences(text):
    text = '\n'.join(text)
    references = 0
    refbegin = 0
    for line in text.split('\n'):
        refbegin += 1
        if "References" in line:
            break
    for line in text.split('\n')[refbegin:]:
        if re.search(r'\d\.$', line.rstrip('\n')):
            references += 1
    return references

def parsevec(path, pdf, pagenum):
    figures = [0]*10
    tables = [0]*10
    equations = [0]*10
    for page in range(min(pagenum, 10)):
        with pdfplumber.open(path) as pdfexc:
            table_page = pdfexc.pages[page]
            tables[page] = len(table_page.extract_tables())
        context = pdf.getPage(page)
        text = context.extractText()
        equations[page] = get_formula_num(text)
        pattern = re.compile(r'(Figure|Table|Fig\.)\s*(\d+)(:)')
        res = pattern.findall(text)
        res = set([''.join(i) for i in res])
        for item in res:
            if item.startswith('Fig'):
                figures[page] += 1
        if figures[page] == 0:
            s = set()
            pattern = re.compile(r'(Figure|Table|Fig\.)\s*(\d+)(\.[A-Z]{0,1})')
            res = pattern.findall(text)
            res = set([''.join(i).replace('.', ':').split(':')[0] for i in res]) - s
            s = s | res
            for item in res:
                if item.startswith('Fig'):
                    figures[page] += 1

    return figures, tables, equations


        
if __name__ == "__main__":
    pdfFileObject = open(pdf, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
    count = pdfReader.numPages
    # page-wise feature
    figures, tables, formulas = parsevec(pdf, pdfReader, count)
    raw = parser.from_file(pdf)
    text = str(raw['content']).split('\n')
    title = word_tokenize(text[0].rstrip('\n'))
    # paper-wise feature
    wordf = getWordsf(title, text[1:])
    references = countReferences(text)