import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CLASSIFIERS = {
    'wine': {
        'fetcher': 'normalize.wine.WineFetcher',
        'normalizer': 'normalize.wine.WineNormalizer',
        'learner': 'learn.wine.WineLearn',
    }
}
