from nltk.classify import ClassifierI
from statistics import mode, StatisticsError


class VoteClassifier(ClassifierI):
    def __init__(self, classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        ret = "neg"
        try:
            ret = mode(votes)
        except StatisticsError:
            # print("Caught1")
            pass
        return ret

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        try:
            choice_votes = votes.count(mode(votes))
            conf = choice_votes / len(votes)
            return conf
        except StatisticsError:
            # print("Caught2")
            return 0.5
