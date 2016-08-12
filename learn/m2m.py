import numpy
import random
from sklearn import svm, cross_validation, linear_model, metrics
from matplotlib import pyplot as plt
from ggplot import qplot


class M2MLearn(object):

    def __init__(self, gamma=0.001, C=100):
        self.gamma = gamma  # aproximate optimial is 0.001
        self.C = C  # Optmial is 100
        self.model = None

    def fit(self, data, target):
        self.model = svm.SVC(gamma=self.gamma, C=self.C, probability=True)
        self.model.fit(data, target)

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
        plt.scatter(
            test_column,
            numpy.array(range(len(test_column))),
            c=test_targets,
            cmap=plt.cm.Paired
        )
        plt.show()
        return test_column, test_targets

    def cross_validate(self, data, target):
        gamma = [0.0001, 0.001, 0.01, 0.1, 1]
        for g_val in gamma:
            tmp_model = svm.SVC(gamma=g_val, C=self.C, probability=True)
            scores = cross_validation.cross_val_score(
                tmp_model, data, target[:, 0], cv=5
            )
            print("Accuracy for %s: %0.2f (+/- %0.2f)" % (str(g_val), scores.mean(), scores.std() * 2))

        C = [0.1, 1, 10, 100, 1000]

        for c_val in C:
            tmp_model = svm.SVC(gamma=self.gamma, C=c_val, probability=True)
            scores = cross_validation.cross_val_score(
                tmp_model, data, target[:, 0], cv=5
            )
            print("Accuracy for %s: %0.2f (+/- %0.2f)" % (str(c_val), scores.mean(), scores.std() * 2))

    def predict(self, datum):
        if not self.model:
            raise Exception('Must train model before predicting')

        return self.model.predict_proba(datum)

    def roc_curve(self, probabilitites, target):
        return metrics.roc_curve(
            target[:, 0],
            probabilitites[:, 0],
            pos_label='1',
        )

    def score(self, datum, target):
        if not self.model:
            raise Exception('Must train model before predicting')

        return self.model.score(datum, target)


class DummyM2MLearn(object):

    def __init__(self, **kwargs):
        self.model = None

    def fit(self, data, target):
        # Finds the probability that a person is a defaulter
        self.model = sum(map(int, target[:, 0])) / float(len(target[:, 0]))
        print self.model

    def score(self, datum, target):
        correct = 0
        for t in target[:, 0]:
            guess = '1' if random.random() <= self.model else '0'
            if guess == t:
                correct += 1

        return correct / float(len(target[:, 0]))

    def predict(self, datum):
        pass
