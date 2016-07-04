from abc import ABCMeta
from collections import namedtuple


class Fetcher(object):
    __metaclass__ = ABCMeta

    def data():
        """Returns an iterator to iterate over the dataset"""
        raise NotImplementedError()

    def targets():
        """Returns an iterator to iterate over the labels"""
        raise NotImplementedError()


class Normalizer(object):

    @classmethod
    def normalize():
        """Returns an iterator to iterate over the labels"""
        raise NotImplementedError()


Dataset = namedtuple('Dataset', ['train', 'test', 'train_targets', 'test_targets'])
