import os

from sklearn import preprocessing

from settings import PROJECT_ROOT
from .abstract import Normalizer, Dataset, Fetcher
from .utils import csv_fetch

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
        return csv_fetch(
            self.data_filepath,
            self.columns,
            self.target_column,
            self.pct_train,
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
