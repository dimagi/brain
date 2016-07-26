import importlib

from invoke import task

from settings import CLASSIFIERS


@task
def run(ctx, classifier=''):
    if not classifier:
        print 'Must specify classifier: invoke run <classifier>'
        return

    classifier_conf = CLASSIFIERS[classifier]
    fetcher = to_class(classifier_conf['fetcher'])()
    normalizer = to_class(classifier_conf['normalizer'])
    normalized_dataset = normalizer.normalize(fetcher.dataset())

    learner = to_class(classifier_conf['learner'])()
    learner.fit(normalized_dataset.train, normalized_dataset.train_targets)

    score = learner.score(normalized_dataset.test, normalized_dataset.test_targets)
    print score


@task
def optimize(ctx, classifier=''):
    if not classifier:
        print 'Must specify classifier: invoke run <classifier>'
        return

    classifier_conf = CLASSIFIERS[classifier]
    fetcher = to_class(classifier_conf['fetcher'])()
    normalizer = to_class(classifier_conf['normalizer'])
    normalized_dataset = normalizer.normalize(fetcher.dataset())

    learner = to_class(classifier_conf['learner'])()
    learner.cross_validate(normalized_dataset.train, normalized_dataset.train_targets)


def to_class(class_dot_path):
    module_path, cls_name = class_dot_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    return cls


@task
def relevance(ctx, classifier='', column=''):
    if not classifier:
        print 'Must specify classifier: invoke relevance <classifier> <column>'
        return
    if not column:
        print 'Must specify column: invoke relevance <classifier> <column>'
        return

    classifier_conf = CLASSIFIERS[classifier]
    fetcher = to_class(classifier_conf['fetcher'])()
    normalizer = to_class(classifier_conf['normalizer'])
    normalized_dataset = normalizer.normalize(fetcher.dataset())

    learner = to_class(classifier_conf['learner'])()
    learner.relevance(
        normalized_dataset,
        column,
    )
