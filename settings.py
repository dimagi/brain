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
    }
}
