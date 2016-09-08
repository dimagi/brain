import numpy
import random
from collections import defaultdict
from sklearn import svm, cross_validation, linear_model, metrics
from matplotlib import pyplot as plt
from ggplot import qplot
from normalize import utils


class M2MLearn(object):

    def __init__(self, gamma=0.001, C=1000):
        self.gamma = gamma  # aproximate optimial is 0.001
        self.C = C  # Optmial is 100
        self.model = None

    def fit(self, data, target):
        raise NotImplementedError()

    def relevance(self, dataset, column):
        regression = linear_model.LinearRegression()
        idx = list(dataset.columns).index(column)

        def reshape(data):
            return numpy.reshape(data, (-1, 1))

        regression.fit(
            reshape(dataset.train[:, idx]).astype(numpy.float),
            dataset.train_targets.astype(numpy.float),
        )
        test_column = reshape(dataset.test[:, idx]).astype(numpy.float)
        test_targets = dataset.test_targets.astype(numpy.float)
        print regression.score(
            test_column,
            test_targets,
        )

        # Graph feature scattered with red dots that indicate LTFU
        # plt.scatter(
        #     numpy.array(range(len(test_column))),
        #     utils.jitter(test_column),
        #     c=test_targets,
        #     cmap=plt.cm.Paired
        # )
        # plt.show()

        values = self._get_bucket_percentages(dataset.train, dataset.train_targets)
        feature_labels = map(lambda value: value[0], values)
        percentages = map(lambda value: value[1], values)
        plt.bar(
            feature_labels,
            percentages,
        )
        plt.title(column)
        plt.axis([0, len(feature_labels), 0, 1])
        plt.show()

        return test_column, test_targets

    def _get_bucket_percentages(self, train, targets):
        EPSILON = 0.01
        buckets = defaultdict(lambda: 0)
        for idx, row in enumerate(train):
            assert len(row) == 1
            buckets[row[0]] += 1

        for key, count in buckets.iteritems():
            buckets[key] = (key, buckets[key] / float(len(targets)))

        assert 1.0 - EPSILON <= sum(map(lambda v: v[1], buckets.values())) <= 1.0 + EPSILON
        return buckets.values()

    def predict(self, datum):
        if not self.model:
            raise Exception('Must train model before predicting')

        return self.model.predict_proba(datum.astype('float'))

    def roc_curve(self, probabilitites, target):
        return metrics.roc_curve(
            target[:, 0],
            probabilitites[:, 0],
            pos_label='1',
        )

    def score_probs(self, y_true, y_prob):
        return metrics.brier_score_loss(y_true, y_prob)

    def score(self, datum, target):
        if not self.model:
            raise Exception('Must train model before predicting')

        return self.model.score(datum, target)


class M2MLearnLogisticRegression(M2MLearn):

    def fit(self, data, target):
        self.model = linear_model.LogisticRegressionCV(
            Cs=self.C,
            scoring=utils.brier_scorer,
            penalty='l2',
            solver='liblinear',
            refit=True,
            n_jobs=-1,
        )
        self.model.fit(data, target)

    def cross_validate(self, data, target):
        pass


class M2MLearnSvm(M2MLearn):

    def fit(self, data, target):
        self.model = svm.SVC(
            gamma=self.gamma,
            C=self.C,
            class_weight={'0': 0.9, '1': 0.1},
            probability=True,
        )
        self.model.fit(data, target)

    def cross_validate(self, data, target):
        gamma = [0.0001, 0.001, 0.01, 0.1, 1]
        for g_val in gamma:
            tmp_model = svm.SVC(gamma=g_val, C=self.C, probability=True)
            scores = cross_validation.cross_val_score(
                tmp_model, data, target[:, 0], cv=5, scoring=utils.brier_scorer
            )
            print("Accuracy for %s: %0.2f (+/- %0.2f)" % (str(g_val), scores.mean(), scores.std() * 2))

        C = [0.1, 1, 10, 100, 1000]

        for c_val in C:
            tmp_model = svm.SVC(gamma=self.gamma, C=c_val, probability=True)
            scores = cross_validation.cross_val_score(
                tmp_model, data, target[:, 0], cv=5, scoring=utils.brier_scorer
            )
            print("Accuracy for %s: %0.2f (+/- %0.2f)" % (str(c_val), scores.mean(), scores.std() * 2))


class DummyM2MLearn(object):

    def __init__(self, **kwargs):
        self.model = None

    def fit(self, data, target):
        # Finds the probability that a person is a defaulter
        self.model = sum(map(int, target[:, 0])) / float(len(target[:, 0]))

    def score(self, datum, target):
        correct = 0
        for t in target[:, 0]:
            guess = '0'
            if guess == t:
                correct += 1

        return correct / float(len(target[:, 0]))

    def score_probs(self, y_true, y_prob):
        return metrics.brier_score_loss(y_true, y_prob)

    def predict(self, datum):
        predictions = map(lambda _: [1 - self.model, self.model], datum)
        return numpy.array(predictions)
