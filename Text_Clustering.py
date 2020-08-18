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
from sklearn.cluster import KMeans

data = pd.read_csv("data//data_processed.csv")
data = data[:8000]

stopwords = [line.strip() for line in open('data//stopwords.txt', 'r', encoding='utf-8').readlines()]
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

count = CountVectorizer(tokenizer=cut_and_remove_stop_words)
countvector = count.fit_transform(data.iloc[:,1]).toarray()

np.save("tf-idf.npy",countvector)

clf = KMeans(n_clusters=400)
s = clf.fit(countvector)

np.save("cluster_result.npy",clf.labels_)
