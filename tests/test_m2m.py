from datetime import datetime
from unittest import TestCase
from mock import patch

from normalize.m2m import M2MNormalizer


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
