import numpy
import random
from sklearn import svm, cross_validation, linear_model, metrics
from ggplot import qplot


class M2MLearn(object):

    def __init__(self, gamma=0.001, C=100):
        self.gamma = gamma  # aproximate optimial is 0.001
        self.C = C  # Optmial is 100
        self.model = None

    def fit(self, data, target):
        self.model = svm.SVC(gamma=self.gamma, C=self.C, probability=True)
        self.model.fit(data, target)

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
        self.model = sum(map(int, target[:, 0])) / len(target[:, 0])

    def score(self, datum, target):
        correct = 0
        for t in target[:, 0]:
            guess = '1' if random.random() <= self.model else '0'
            if guess == t:
                correct += 1

        return correct / float(len(target[:, 0]))
