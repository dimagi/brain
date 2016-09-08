import csv
import random
from operator import itemgetter
from sklearn import metrics
from matplotlib import pyplot as plt

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
            value = getter(row)
            if not isinstance(value, tuple):
                value = [value]

            if random.random() <= pct_train:
                train = numpy.append(train, value, axis=0)
                train_targets = numpy.append(train_targets, [row[target_column]], axis=0)
            else:
                test = numpy.append(test, value, axis=0)
                test_targets = numpy.append(test_targets, [row[target_column]], axis=0)

    return Dataset(
        target_column=target_column,
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


def brier_scorer(estimator, X, y):
    probabilities = estimator.predict_proba(X)
    return metrics.brier_score_loss(
        map(lambda d: float(d), y),
        probabilities[:, 1],
    )


def heatmap(y_true, y_prob):
    """
    Outputs a heatmap for the given y_prob

    :param: y_true - an array of truths (nparray)
    :param: y_prob - an array of probabilities (nparray)
    """
    if len(y_prob.shape) == 1:
        # Convert to 1d matrix
        y_prob = numpy.reshape(y_prob, newshape=(y_prob.shape[0], 1))

    max_prob = max(y_prob[:, 0])

    if len(y_true.shape) == 1:
        # Convert to 1d matrix
        y_true = numpy.reshape(y_true, newshape=(y_true.shape[0], 1))

    max_trues = numpy.reshape(
        map(lambda x: min(x, max_prob), y_true[:, 0]),
        newshape=y_true.shape,
    )
    data = numpy.append(y_prob, max_trues, axis=1)

    plt.pcolor(data, cmap=plt.cm.Blues)
    plt.show()


def jitter(array):
    """
    Adds jitter to an array value for better viewing
    """
    return map(
        lambda x: x + (random.random() * 0.2) - 0.1,
        array,
    )
