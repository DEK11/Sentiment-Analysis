from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from nltk import NaiveBayesClassifier, classify
from gsu.data import TestTrainData
import pickle


def TrainClassifiers():
    training_set, testing_set = TestTrainData()

    classifiers = list()
    classifier_name = list()

    NaiveBayesClassifier_classifier = NaiveBayesClassifier.train(training_set)
    classifiers.append(NaiveBayesClassifier_classifier)
    classifier_name.append("NaiveBayesClassifier")

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    classifiers.append(MNB_classifier)
    classifier_name.append("MultinomialNBClassifier")

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    classifiers.append(BernoulliNB_classifier)
    classifier_name.append("BernoulliNBClassifier")

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    classifiers.append(LogisticRegression_classifier)
    classifier_name.append("LogisticRegressionClassifier")

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    classifiers.append(LogisticRegression_classifier)
    classifier_name.append("LinearSVCClassifier")

    SGDC_classifier = SklearnClassifier(SGDClassifier())
    SGDC_classifier.train(training_set)
    classifiers.append(SGDC_classifier)
    classifier_name.append("SGDClassifier")

    print("Naive_Bayes Algo accuracy percent:", (classify.accuracy(NaiveBayesClassifier_classifier, testing_set))*100)
    print("MNB_classifier accuracy percent:", (classify.accuracy(MNB_classifier, testing_set))*100)
    print("BernoulliNB_classifier accuracy percent:", (classify.accuracy(BernoulliNB_classifier, testing_set))*100)
    print("LogisticRegression_classifier accuracy percent:", (classify.accuracy(LogisticRegression_classifier, testing_set))*100)
    print("LinearSVC_classifier accuracy percent:", (classify.accuracy(LinearSVC_classifier, testing_set))*100)
    print("SGDClassifier accuracy percent:", (classify.accuracy(SGDC_classifier, testing_set))*100)

    SaveClassifiers(classifiers, classifier_name)

    return classifiers


def SaveClassifiers(classifiers, classifier_name):

    for i in range(0, len(classifiers)):
        save_classifier_path = open("saved/" + classifier_name[i] + ".p", "wb")
        pickle.dump(classifiers[i], save_classifier_path)
        save_classifier_path.close()

    save_classifier_path = open("saved/classifier_name.p", "wb")
    pickle.dump(classifier_name, save_classifier_path)

