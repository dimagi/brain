import os
from dateutil.parser import parse
from datetime import datetime

from sklearn import preprocessing

from settings import PROJECT_ROOT
from .abstract import Fetcher, Normalizer, Dataset
from .utils import csv_fetch

LTFU_DAYS = 10


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
        for idx, column in enumerate(dataset.columns):
            dataset.train[:, idx] = getattr(cls, column, cls.default_normalize)(dataset.train[:, idx])
            dataset.test[:, idx] = getattr(cls, column, cls.default_normalize)(dataset.test[:, idx])

        dataset.train_targets[:, 0] = getattr(cls, 'target')(dataset.train_targets[:, 0])
        dataset.test_targets[:, 0] = getattr(cls, 'target')(dataset.test_targets[:, 0])

        return dataset

    @staticmethod
    def default_normalize(column_data):
        return M2MNormalizer._labeled_data(column_data)

    @staticmethod
    def target(column_data):
        def classify(date):
            try:
                return int((parse(date) - datetime.today()).days > LTFU_DAYS)
            except ValueError:
                return 0

        return map(
            classify,
            column_data,
        )

    @staticmethod
    def age(column_data):
        def classify(age):
            if int(age) < 25:
                return 'young'
            elif int(age) >= 25 or int(age) < 50:
                return 'middle-age'
            else:
                return 'old'
        return M2MNormalizer._labeled_data(map(classify, column_data))

    @staticmethod
    def _labeled_data(column_data):
        le = preprocessing.LabelEncoder()
        le.fit(column_data)
        return le.transform(column_data)
