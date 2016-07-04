from sklearn import svm


class WineLearn(object):

    def __init__(self, gamma=0.001, C=100):
        self.gamma = gamma
        self.C = C
        self.model = None

    def fit(self, data, target):
        self.model = svm.SVC(gamma=self.gamma, C=self.C)
        self.model.fit(data, target)

    def predict(self, datum):
        if not self.model:
            raise Exception('Must train model before predicting')

        return self.model.predict(datum)
