import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CLASSIFIERS = {
    'wine': {
        'fetcher': 'normalize.wine.WineFetcher',
        'normalizer': 'normalize.wine.WineNormalizer',
        'learner': 'learn.wine.WineLearn',
    },
    'spf-defaulters': {
        'fetcher': 'normalize.spf_defaulters.SpfDefaultersFetcher',
        'normalizer': 'normalize.spf_defaulters.SpfDefaultersNormalizer',
        'learner': 'learn.spf_defaulters.SpfDefaultersLearn',
    },
    'dummy-spf-defaulters': {
        'fetcher': 'normalize.spf_defaulters.SpfDefaultersFetcher',
        'normalizer': 'normalize.spf_defaulters.SpfDefaultersNormalizer',
        'learner': 'learn.spf_defaulters.DummySpfDefaultersLearn',
    },
    'm2m': {
        'fetcher': 'normalize.m2m.M2MFetcher',
        'normalizer': 'normalize.m2m.M2MNormalizer',
        'learner': 'learn.m2m.M2MLearn',
    },
    'dummy-m2m': {
        'fetcher': 'normalize.m2m.M2MFetcher',
        'normalizer': 'normalize.m2m.M2MNormalizer',
        'learner': 'learn.m2m.DummyM2MLearn',
    },
}
