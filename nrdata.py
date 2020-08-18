import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
import pandas as pd

from utils import OrderedCounter, cut_words


class NRData(Dataset):

    def __init__(self, data_dir, split, create_data, cluster_result_dir, N, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.cluster_result = np.load(cluster_result_dir)
        self.N = N

        self.max_news_length = kwargs.get('max_news_length', 50)
        self.max_report_length = kwargs.get('max_report_length', 150)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, 'data_' + split + '.txt')
        self.data_file = 'data_' + split + '.json'
        self.vocab_file = 'data_vocab.json'

        if create_data:
            print("Creating new %s data." % split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new." % (
            split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'news': np.asarray(self.data[idx]['news']),
            'report': np.asarray(self.data[idx]['report']),
            'input_report': np.asarray(self.data[idx]['input_report']),
            'news_length': self.data[idx]['news_length'],
            'report_length': self.data[idx]['report_length'],
            'N_news':np.asarray(self.data[idx]['N_news']),
            'N_reports': np.asarray(self.data[idx]['N_reports']),
            'N_reports_length':self.data[idx]['N_reports_length'],
            'N_news_length': self.data[idx]['N_news_length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        data = defaultdict(dict)

        csv_file = pd.read_csv(self.raw_data_path)
        for index, row in csv_file.iterrows():
            news, report = row['news'], row['reports']
            news = cut_words(news)
            report = cut_words(report)

            label = self.cluster_result[index]

            others = np.where(self.cluster_result == label)[0]
            N_news = []
            N_reports = []

            for i in others:
                N_news += csv_file.iloc[i]['news']+['<sep>']
                N_reports += csv_file.iloc[i]['reports']+['<sep>']

            N_news = N_news[:-1]
            N_reports = N_reports[:-1]

            N_news_length = len(N_news)
            N_reports_length = len(N_reports)


            N_news.extend(['<pad>'] * (self.N * self.max_news_length - N_news_length))
            N_reports(['<pad>'] * (self.N * self.max_report_length - N_reports_length))

            N_news = [self.w2i.get(w, self.w2i['<unk>']) for w in N_news]
            N_reports = [self.w2i.get(w, self.w2i['<unk>']) for w in N_reports]

            # news = ['<sos>'] + news
            news = news[:self.max_news_length]

            input_report = ['<sos>'] + report
            input_report = news[:self.max_report_length]

            report = report[:self.max_report_length - 1]
            report = report + ['<eos>']

            news_length = len(news)
            report_length = len(report)

            news.extend(['<pad>'] * (self.max_news_length - news_length))
            report.extend(['<pad>'] * (self.max_report_length - report_length))

            news = [self.w2i.get(w, self.w2i['<unk>']) for w in news]
            report = [self.w2i.get(w, self.w2i['<unk>']) for w in report]

            id = len(data)
            data[id]['news'] = news
            data[id]['report'] = report
            data[id]['input_report'] = input_report
            data[id]['N_news'] = N_news ##
            data[id]['N_reports'] = N_reports ##
            data[id]['input_report'] = input_report
            data[id]['news_length'] = news_length
            data[id]['N_reports_length'] = N_reports_length ##
            data[id]['N_news_length'] = N_news_length ##

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>', '<num>','<sep>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        data = pd.read_csv(self.raw_data_path)
        for index, row in data.iterrows():
            news, report = row['news'], row['reports']
            news = cut_words(news)
            report = cut_words(report)
            w2c.update(news + report)

        for w, c in w2c.items():
            if c > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)
        print('************')
        print("Vocablurary of %i keys created." % len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
