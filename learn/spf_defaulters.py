import random
from sklearn import svm


class SpfDefaultersLearn(object):

    def __init__(self, gamma=0.001, C=100):
        self.gamma = gamma
        self.C = C
        self.model = None

    def fit(self, data, target):
        self.model = svm.SVC(gamma=self.gamma, C=self.C, probability=True)
        self.model.fit(data, target)

    def cross_validate(self, data, target):
        pass

    def predict(self, datum):
        if not self.model:
            raise Exception('Must train model before predicting')

        return self.model.predict(datum)

    def score(self, datum, target):
        if not self.model:
            raise Exception('Must train model before predicting')

        return self.model.score(datum, target)


class DummySpfDefaultersLearn(object):

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
