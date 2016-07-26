import os
import csv
import random
from operator import itemgetter

import numpy
from sklearn import preprocessing

from settings import PROJECT_ROOT
from .abstract import Fetcher, Normalizer, Dataset

MISSING = '---'


class SpfDefaultersFetcher(Fetcher):
    """

    Place the patient case data in csv file in data/spf-defaulters/patient_data.csv
    """

    data_filepath = os.path.join(PROJECT_ROOT, 'data', 'spf-defaulters', 'patient_data.csv')
    target_column = 'defaulted'
    columns = [
        'cd4_count',
        'health_status',
        'high_risk',
        'hiv_status',
        'moved_away',
        'next_appt_type_pmtct',
        'village',
        'current_status',
        'late',
        'given_birth',
        'contacted',
    ]

    def __init__(self, pct_train=0.8):
        self.pct_train = pct_train

    def dataset(self):
        train = numpy.array([])
        test = numpy.array([])
        train_targets = numpy.array([])
        test_targets = numpy.array([])
        getter = itemgetter(*self.columns)

        with open(self.data_filepath, 'rb') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if random.random() <= self.pct_train:
                    train = numpy.append(train, getter(row), axis=0)
                    train_targets = numpy.append(train_targets, [row[self.target_column]], axis=0)
                else:
                    test = numpy.append(test, getter(row), axis=0)
                    test_targets = numpy.append(test_targets, [row[self.target_column]], axis=0)

        return Dataset(
            columns=numpy.array(self.columns),
            test=numpy.reshape(test, (-1, len(self.columns))),
            train=numpy.reshape(train, (-1, len(self.columns))),
            test_targets=numpy.reshape(test_targets, (-1, 1)),
            train_targets=numpy.reshape(train_targets, (-1, 1)),
        )


class SpfDefaultersNormalizer(Normalizer):

    @classmethod
    def normalize(cls, dataset):
        for idx, column in enumerate(dataset.columns):
            if hasattr(cls, column):
                dataset.train[:, idx] = getattr(cls, column)(dataset.train[:, idx])
                dataset.test[:, idx] = getattr(cls, column)(dataset.test[:, idx])

        return dataset

    @staticmethod
    def cd4_count(column_data):
        return preprocessing.scale(map(
            lambda datum: int(datum) if datum != MISSING and datum else -1,
            column_data,
        ))

    @staticmethod
    def high_risk(column_data):
        return map(int, column_data)

    @staticmethod
    def hiv_status(column_data):
        return map(
            lambda datum: -1 if datum == 'negative' else 1,
            column_data
        )

    @staticmethod
    def moved_away(column_data):
        return map(int, column_data)

    @staticmethod
    def health_status(column_data):
        return SpfDefaultersNormalizer._labeled_data(column_data)

    @staticmethod
    def given_birth(column_data):
        return SpfDefaultersNormalizer._labeled_data(column_data)

    @staticmethod
    def next_appt_type_pmtct(column_data):
        return SpfDefaultersNormalizer._labeled_data(column_data)

    @staticmethod
    def village(column_data):
        return SpfDefaultersNormalizer._labeled_data(column_data)

    @staticmethod
    def current_status(column_data):
        return SpfDefaultersNormalizer._labeled_data(column_data)

    @staticmethod
    def late(column_data):
        return map(int, column_data)

    @staticmethod
    def _labeled_data(column_data):
        le = preprocessing.LabelEncoder()
        le.fit(column_data)
        return le.transform(column_data)
