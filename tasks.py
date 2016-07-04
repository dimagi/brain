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
    predictions = learner.predict(normalized_dataset.test)

    correct = 0
    wrong = 0
    for idx, prediction in enumerate(predictions):
        if prediction == normalized_dataset.test_targets[idx]:
            correct += 1
        else:
            wrong += 1

    print 'You got {} correct! and {} wrong!'.format(correct, wrong)


def to_class(class_dot_path):
    module_path, cls_name = class_dot_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    return cls