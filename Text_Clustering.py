import jieba
import pandas as pd
import numpy as np
import os
import itertools 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

text = "2018年11月30日，公司发布《关于回购部分社会公众股的预案的公告》，宣布将在不超过12个月的时间内，以不超过人民币8.00元/股的价格回购资金总额不少于人民币6亿元，不超过人民币12亿的股票，公司表示，拟回购股份将用于《中华人民共和国公司法（2018年修订）》中第一百四十二条第(三)项、第(五)项规定的用途,生猪养殖业务加速发展，公司业绩拐点已至通过对历史上所有生猪养殖项目进行梳理，我们发现自16年之后，公司猪场建设进度显著加快按照公司从动工到最终出栏需要两年的时间计算，我们认为2017年底公司将有望形成700万头生猪产能，将在2020年进行释放，加上新六模式生猪出栏量，我们预计公司2020年将出栏生猪800-1000万头假设均价和成本分别为12.6/15/16元/kg和12.6/12.5/12.4元/kg，综合体重100kg，对应贡献利润0/9.75/36.8亿，"
corpus=["我 来到 北京 清华大学",#第一类文本切词后的结果，词之间以空格隔开
		"他 来到 了 网易 杭研 大厦",#第二类文本的切词结果
		"小明 硕士 毕业 与 中国 科学院",#第三类文本的切词结果
		"我 爱 北京 天安门"]

stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()]
stopwords.append('\n')
numbers = [str(i) for i in range(10)]

def cut_and_remove_stop_words(sentence):
    words = jieba.cut(sentence, cut_all=False)
    result = []

    for word in words:
        has_num = False
        for num in numbers:
            if num in word:
                has_num = True
                break 
        if word not in stopwords and not has_num:
            result.append(word)

    return result

# print(cut_and_remove_stop_words(text))
count = CountVectorizer(tokenizer=cut_and_remove_stop_words)
countvector = count.fit_transform(corpus)
print(count.get_feature_names())
