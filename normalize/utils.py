import csv
import random
from operator import itemgetter
import numpy

from .abstract import Dataset


def csv_fetch(data_filepath, columns, target_column, pct_train=0.8):
    train = numpy.array([])
    test = numpy.array([])
    train_targets = numpy.array([])
    test_targets = numpy.array([])

    exploded_columns = _explode_columns(columns)
    getter = itemgetter(*exploded_columns)

    with open(data_filepath, 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if random.random() <= pct_train:
                train = numpy.append(train, getter(row), axis=0)
                train_targets = numpy.append(train_targets, [row[target_column]], axis=0)
            else:
                test = numpy.append(test, getter(row), axis=0)
                test_targets = numpy.append(test_targets, [row[target_column]], axis=0)

    return Dataset(
        columns=numpy.array(exploded_columns),
        test=numpy.reshape(test, (-1, len(exploded_columns))),
        train=numpy.reshape(train, (-1, len(exploded_columns))),
        test_targets=numpy.reshape(test_targets, (-1, 1)),
        train_targets=numpy.reshape(train_targets, (-1, 1)),
    )


def _explode_columns(columns):
    exploded_columns = []
    for column in columns:
        if isinstance(column, basestring):
            exploded_columns.append(column)
        else:
            # Call exploding function
            exploded_columns.extend(column[1](column[0]))
    return exploded_columns
