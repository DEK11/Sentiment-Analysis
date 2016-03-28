# from gsu.train import TrainClassifiers
from gsu.load import LoadClassifiers
from gsu.load import LoadFeatures
from gsu.data import find_features
from gsu.VoteClassifier import VoteClassifier


class Sentiment:

    def __init__(self):
        # classifiers = TrainClassifiers()
        classifiers = LoadClassifiers()
        self.votedClassifier = VoteClassifier(classifiers)
        self.new_features = LoadFeatures()

    def Analyse(self, text):
        new_features = find_features(text, self.new_features)
        return self.votedClassifier.classify(new_features), self.votedClassifier.confidence(new_features)
