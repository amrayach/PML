#!/usr/bin/env python3
from collections import Counter
import random

from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd
import torch
import json
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import utils
import configparser
from sklearn.model_selection import KFold




def get_sample_weights(labels):
    counter = Counter(labels)
    counter = dict(counter)
    for k in counter:
        counter[k] = 1 / counter[k]
    sample_weights = np.array([counter[l] for l in labels])
    return sample_weights


def load_data(args, mode='train'):
    usecols = list(map(lambda x: int(x), args.get('Data', 'usecols').split(',')))
    chunks = pd.read_csv(args.get('Data', 'dataset') + '/' + mode + '.csv',
                         usecols=usecols,
                         chunksize=args.getint('Data', 'chunk_size'),
                         encoding=args['Data'].get('encoding'),
                         nrows=args.getint('Data', 'max_csv_rows'),
                         sep=args['Data'].get('csv_sep'),
                         doublequote=True)


    texts = []
    labels = []
    for chunk in tqdm(chunks):
        curr_chunk = chunk.copy()
        curr_chunk = curr_chunk.sample(frac=1)
        # maybe concat with new line instead of space
        if args.get('Data', 'dataset') == 'ag_news':
            curr_chunk['merged_text'] = curr_chunk.iloc[:, 1] + ' ' + curr_chunk.iloc[:, 2]
            texts += curr_chunk['merged_text'].tolist()
        else:
            texts += curr_chunk.iloc[:, 1].tolist()

        labels += curr_chunk.iloc[:, 0].tolist()

    # pre-processing data
    if bool(args.getboolean('Data', 'preprocess_data')):
        texts = list(map(lambda text: utils.process_text(args.get('Data', 'steps').split(','), text), texts))

    # class balancing
    if bool(args.getboolean('Data', 'balance_classes')):

        counter = Counter(labels)
        keys = list(counter.keys())
        values = list(counter.values())
        count_minority = np.min(values)

        balanced_labels = []
        balanced_texts = []

        for key in keys:
            balanced_texts += [text for text, label in zip(texts, labels) if label == key][
                              :int(args.getint('Data', 'ratio') * count_minority)]
            balanced_labels += [label for text, label in zip(texts, labels) if label == key][
                               :int(args.getint('Data', 'ratio') * count_minority)]

        texts = balanced_texts
        labels = balanced_labels

    #if args.get('Data', 'dataset') == 'yelp':
    #    labels = list(map(lambda x: 1 if x in [1,2] else (3 if x in [4,5] else 2), labels))

    number_of_classes = len(set(labels))
    sample_weights = get_sample_weights(labels)

    if args.getint('Data', 'k_folds') > 1:
        text_label_list = np.array(list(zip(texts, labels)))
        np.random.shuffle(text_label_list)
        kf = KFold(n_splits=args.getint('Data', 'k_folds'), shuffle=True)
        folds = []
        for train_index, test_index in kf.split(text_label_list):
            train_texts, train_labels = zip(*list(text_label_list[train_index]))
            test_texts, test_labels = zip(*list(text_label_list[test_index]))
            train_texts = list(train_texts)
            train_labels = list(map(lambda x: int(x), list(train_labels)))
            train_weights = get_sample_weights(train_labels)

            test_texts = list(test_texts)
            test_labels = list(map(lambda x: int(x), list(test_labels)))
            test_weights = get_sample_weights(test_labels)
            #res = (train_texts, torch.LongTensor(train_labels), train_weights, test_texts, torch.LongTensor(test_labels), test_weights, number_of_classes)
            folds.append((train_texts, torch.LongTensor(train_labels), train_weights, test_texts, torch.LongTensor(test_labels), test_weights, number_of_classes))
        return folds



    return texts, torch.LongTensor(labels), number_of_classes, sample_weights


class MyDataset(Dataset):
    def __init__(self, texts, labels, args):
        self.data = texts
        self.labels = labels
        self.length = len(self.labels)

        self.alphabet = args.get('DataSet', 'alphabet')
        self.number_of_characters = args.getint('DataSet', 'char_num')

        self.l0 = args.getint('DataSet', 'l0')
        self.identity_mat = np.identity(self.number_of_characters)

    def __len__(self):
        return self.length


    def __getitema__(self, idx):
        raw_text = self.data[idx]

        data = np.array([self.identity_mat[self.alphabet.index(i)] for i in list(raw_text)[::-1] if i in self.alphabet],
                        dtype=np.float32)
        if len(data) > self.l0:
            data = data[:self.l0]
        elif 0 < len(data) < self.l0:
            data = np.concatenate(
                (data, np.zeros((self.l0 - len(data), self.number_of_characters), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros(
                (self.l0, self.number_of_characters), dtype=np.float32)

        label = self.labels[idx]
        data = torch.Tensor(data)

        return data, label

    def getClassWeight(self):
        num_samples = self.__len__()
        label_set = set(self.labels)
        num_class = [self.labels.count(c) for c in label_set]
        class_weight = [num_samples/float(self.labels.count(c)) for c in label_set]
        return class_weight, num_class


    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.labels[idx]
        return X, y

    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx][:self.l0]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char) != -1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)



if __name__ == '__main__':
    args = configparser.ConfigParser()
    args.read('argsConfig.ini')
    texts, labels, number_of_classes, sample_weights = load_data(args)

    #Dat = MyDataset(texts, labels, args)
    #gen = DataLoader(Dat, batch_size=128, num_workers=1, drop_last=True, shuffle=True)
    #for i_batch, data in enumerate(gen, start=1):
    #    print(i_batch, data)

    print()
