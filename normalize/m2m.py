import os
import numpy
from dateutil.parser import parse
from datetime import datetime

from sklearn import preprocessing
from imblearn.datasets import make_imbalance

from settings import PROJECT_ROOT
from .abstract import Fetcher, Normalizer, Dataset
from .utils import csv_fetch
from ggplot import qplot

LTFU_DAYS = 90
DATE_OF_DOWNLOAD = datetime(2016, 7, 29)  # Date when dataset was fetched


def explode_appointment(prefix):
    """
    Takes a prefix and returns relevant columns
    """
    columns = []

    if prefix.startswith('call'):
        columns.append('{}_outcome'.format(prefix))
    elif prefix.startswith('home_visit'):
        columns.append('{}_outcome'.format(prefix))
        columns.append('{}_outcome_known'.format(prefix))
        columns.append('{}_done_by'.format(prefix))
    elif prefix.startswith('sms'):
        columns.append('{}_outcome'.format(prefix))
        columns.append('{}_done'.format(prefix))
        columns.append('{}_done_date'.format(prefix))
    else:
        columns.append('{}_due_date'.format(prefix))
        columns.append('{}_status'.format(prefix))
        columns.append('{}_done_date'.format(prefix))
    return columns


class M2MFetcher(Fetcher):
    """
    Plase the mother case data in a csv file in data/m2m/mothers.csv
    Other options:

    mothers_small.csv
    mothers_small_tail.csv
    mothers_very_small.csv
    """
    data_filepath = os.path.join(PROJECT_ROOT, 'data', 'm2m', 'mothers_small_tail.csv')
    target_column = 'next_visit_date'
    columns = [
        'age',
        'acfu_status',
        'agree_to_acfu',
        'acfu_most_recent_reached_date',
        'acfu_returned_date',
        'breastfeeding_stopped',
        'client_status',
        'client_type',
        'country',
        'dob',
        'date_of_conception',
        'current_infant_feeding_method',
        'edd',
        'ever_breastfed',
        'facility',
        'gestation_first_anc',
        'gestation_first_m2m',
        'partner_hiv_status',
        'province',
        'number_anc',
        'number_art',
        ('an2', explode_appointment),
        ('an3', explode_appointment),
        ('an4', explode_appointment),
        ('birth_pcr', explode_appointment),
        ('call_1', explode_appointment),
        ('call_2', explode_appointment),
        ('call_3', explode_appointment),
        ('cd4_prev', explode_appointment),
        ('eighteen_twentyfour_month_infant_test', explode_appointment),
        ('family_planning', explode_appointment),
        ('home_visit1', explode_appointment),
        ('home_visit2', explode_appointment),
        ('infant_art_init', explode_appointment),
        # ('infant_art_refill', explode_appointment),
        ('infant_ctx', explode_appointment),
        ('maternal_art_init', explode_appointment),
        # ('mother_hiv_retest', explode_appointment),
        ('nine_month_infant_test', explode_appointment),
        ('pn2', explode_appointment),
        ('six_eight_week_pcr_result', explode_appointment),
        ('sms_1', explode_appointment),
        ('sms_2', explode_appointment),
        ('sms_3', explode_appointment),
        ('tb_test', explode_appointment),
        ('tb_treatment', explode_appointment),
        ('ten_month_infant_test', explode_appointment),
        ('thirteen_month_infant_test', explode_appointment),
        # ('viral_load', explode_appointment),

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


class M2MNormalizer(Normalizer):
    computed_columns = []

    @classmethod
    def normalize(cls, dataset):
        def reshape(data):
            return numpy.reshape(data, (-1, 1))

        idx = 0
        normalized_train = dataset.train
        normalized_test = dataset.test
        normalized_train_targets = dataset.train_targets
        normalized_test_targets = dataset.test_targets

        for column in dataset.columns:
            normalized_train, index_to_add = cls._apply_normalize(normalized_train, column, idx)
            normalized_test, index_to_add = cls._apply_normalize(normalized_test, column, idx)
            idx += index_to_add

        normalized_train_targets, _ = cls._apply_normalize(normalized_train_targets, 'target', 0)
        normalized_test_targets, _ = cls._apply_normalize(normalized_test_targets, 'target', 0)
        return Dataset(
            columns=dataset.columns,
            train=normalized_train,
            test=normalized_test,
            train_targets=normalized_train_targets,
            test_targets=normalized_test_targets,
        )

    @classmethod
    def _apply_normalize(cls, matrix, column, idx):
        normalized_values = getattr(cls, column, cls.default_normalize)(matrix[:, idx])

        # Dealing with an array of arrays
        if isinstance(normalized_values[0], numpy.ndarray):
            matrix = numpy.delete(matrix, idx, axis=1)
            matrix = numpy.insert(
                matrix,
                idx,
                numpy.reshape(
                    normalized_values,
                    list(reversed(normalized_values.shape)),
                ),
                axis=1
            )
            return matrix, len(normalized_values[0])
        else:
            matrix[:, idx] = normalized_values
            return matrix, 1

    @staticmethod
    def default_normalize(column_data):
        return M2MNormalizer._labeled_data(column_data)

    @staticmethod
    def target(column_data):
        def classify(date):
            try:
                return int((DATE_OF_DOWNLOAD - parse(date)).days > LTFU_DAYS)
            except ValueError:
                return 0

        return map(
            classify,
            column_data,
        )

    @staticmethod
    def edd(column_data):
        return map(
            lambda edd: 0 if edd == '---' or not edd else 1,
            column_data,
        )

    @staticmethod
    def date_of_conception(column_data):
        return map(
            lambda d: 0 if d == '---' or not d else 1,
            column_data,
        )

    @staticmethod
    def province(column_data, n_values=10):
        return M2MNormalizer._one_hot_encoder(column_data, n_values)

    @staticmethod
    def country(column_data, n_values=5):
        return M2MNormalizer._one_hot_encoder(column_data, n_values)

    @staticmethod
    def client_status(column_data, n_values=6):
        return M2MNormalizer._one_hot_encoder(column_data, n_values)

    @staticmethod
    def age(column_data):
        one_hot_encoder = preprocessing.OneHotEncoder()
        # <15, 15-24, over 25 are the usual PEPFAR

        def classify(age):
            if int(age) < 16:
                return 0
            elif int(age) >= 16 and int(age) < 25:
                return 1
            elif int(age) >= 25 and int(age) < 30:
                return 2
            elif int(age) >= 30 and int(age) < 35:
                return 3
            elif int(age) >= 35 and int(age) < 40:
                return 4
            else:
                return 5
        mapped_ages = map(lambda age: [age], map(classify, column_data))
        one_hot_encoder.fit(mapped_ages)
        return one_hot_encoder.transform(mapped_ages).toarray()

    @staticmethod
    def _one_hot_encoder(column_data, n_values):
        """
        Converts labeled data to vectorized binaries:

        [
            'apple',
            'bananna',
            'apple'
        ]
        -->
        [
            [1, 0],
            [0, 1],
            [1, 0],
        ]
        """
        one_hot_encoder = preprocessing.OneHotEncoder(n_values=n_values)
        mapped_labels = map(lambda d: [d], M2MNormalizer._labeled_data(column_data))
        one_hot_encoder.fit(mapped_labels)
        return one_hot_encoder.transform(mapped_labels).toarray()

    @staticmethod
    def _labeled_data(column_data):
        le = preprocessing.LabelEncoder()
        le.fit(column_data)
        return le.transform(column_data)
