from datetime import datetime
from unittest import TestCase
from mock import patch
import numpy as np

from normalize.m2m import M2MNormalizer, Dataset


class TestNormalizer(TestCase):

    @patch(
        'normalize.m2m.DATE_OF_DOWNLOAD',
        datetime(2016, 05, 15),
    )
    def test_ltfu_calculation(self):
        column_data = [
            '---',
            '2016-07-28',
            '2015-05-05',  # LTFU
            '',
        ]

        result = M2MNormalizer.target(column_data)
        self.assertItemsEqual(
            result,
            [0, 0, 1, 0],
        )

    def test_age(self):
        column_data = [
            '14',
            '35',
            '56',
        ]
        result = M2MNormalizer.age(column_data)
        expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        for idx, row in enumerate(expected):
            self.assertItemsEqual(result[idx], row)

    def test_edd(self):
        column_data = [
            '---',
            '7/25/16',
            '10/16/16',
        ]
        result = M2MNormalizer.edd(column_data)
        expected = [0, 1, 1]

        self.assertItemsEqual(result, expected)

    def test_province(self):
        column_data = [
            'Western Cape',
            'KwaZulu Natal',
            'KwaZulu Natal',
        ]
        result = M2MNormalizer.province(column_data, 2)
        expected = [
            [1, 0],
            [0, 1],
            [0, 1],
        ]
        print result
        for idx, row in enumerate(expected):
            self.assertItemsEqual(result[idx], row)

    def test_normalize(self):
        dataset = Dataset(
            columns=np.array(['age', 'fruit', 'letter']),
            train=np.array([
                ['14', 'banana', 'a'],
                ['35', 'apple', 'b'],
                ['56', 'pear', 'c'],
            ]),
            test=np.array([
                ['14', 'banana', 'a'],
                ['35', 'apple', 'b'],
                ['56', 'pear', 'c'],
            ]),
            train_targets=np.array([['---'], ['2016-07-28'], ['2015-05-05']]),
            test_targets=np.array([['---'], ['---'], ['---']]),
        )

        normalized = M2MNormalizer.normalize(dataset)
        expected_train = [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 2, 2],
        ]
        expected_train_targets = [
            [0],
            [0],
            [1],
        ]

        for idx, row in enumerate(expected_train):
            self.assertItemsEqual(normalized.train.astype(float)[idx], row)

        for idx, row in enumerate(expected_train_targets):
            self.assertItemsEqual(normalized.train_targets.astype(float)[idx], row)

    def test_normalize_vector_length_one(self):
        dataset = Dataset(
            columns=np.array(['fruit', 'age']),
            train=np.array([
                ['banana', '14'],
                ['apple', '35'],
                ['pear', '56'],
            ]),
            test=np.array([
                ['banana', '14'],
                ['apple', '35'],
                ['pear', '56'],
            ]),
            train_targets=np.array([['---'], ['2016-07-28'], ['2015-05-05']]),
            test_targets=np.array([['---'], ['---'], ['---']]),
        )

        normalized = M2MNormalizer.normalize(dataset)
        expected_train = [
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [2, 0, 0, 1],
        ]
        expected_train_targets = [
            [0],
            [0],
            [1],
        ]

        for idx, row in enumerate(expected_train):
            self.assertItemsEqual(normalized.train.astype(float)[idx], row)

        for idx, row in enumerate(expected_train_targets):
            self.assertItemsEqual(normalized.train_targets.astype(float)[idx], row)
