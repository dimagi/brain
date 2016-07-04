import os
import csv
import random

import numpy
from sklearn import preprocessing

from settings import PROJECT_ROOT
from .abstract import Fetcher, Normalizer, Dataset


class WineFetcher(Fetcher):

    data_filepath = os.path.join(PROJECT_ROOT, 'data', 'wine', 'wine.data')
    n_cols = 13

    def __init__(self, pct_train=0.8):
        self.pct_train = pct_train

    def dataset(self):
        train = numpy.array([])
        test = numpy.array([])
        train_targets = numpy.array([])
        test_targets = numpy.array([])

        with open(self.data_filepath, 'rb') as f:
            wine_reader = csv.reader(f)
            for row in wine_reader:
                if random.random() <= self.pct_train:
                    train = numpy.append(train, row[1:], axis=0)
                    train_targets = numpy.append(train_targets, [row[0]], axis=0)
                else:
                    test = numpy.append(test, row[1:], axis=0)
                    test_targets = numpy.append(test_targets, [row[0]], axis=0)

        return Dataset(
            test=numpy.reshape(test, (-1, self.n_cols)),
            train=numpy.reshape(train, (-1, self.n_cols)),
            test_targets=numpy.reshape(test_targets, (-1, 1)),
            train_targets=numpy.reshape(train_targets, (-1, 1)),
        )


class WineNormalizer(Normalizer):

    @staticmethod
    def normalize(dataset):
        return Dataset(
            test=preprocessing.scale(dataset.test),
            train=preprocessing.scale(dataset.train),
            test_targets=dataset.test_targets,
            train_targets=dataset.train_targets,
        )
