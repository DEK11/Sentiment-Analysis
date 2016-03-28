import random
import pickle
from nltk.tokenize import word_tokenize


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def LoadData(shuffle=False):

    documents_f = open("saved/documents.p", "rb")
    documents = pickle.load(documents_f)
    documents_f.close()

    documents_f = open("saved/word_features5k.p", "rb")
    word_features = pickle.load(documents_f)
    documents_f.close()

    features = [(find_features(rev, word_features), category) for (rev, category) in documents]

    if shuffle:
        random.shuffle(features)

    testing_set = features[10000:]
    training_set = features[:10000]

    return training_set, testing_set


def LoadClassifiers():
    document = open("saved/classifier_name.p", "rb")
    classifier_name = pickle.load(document)
    document.close()
    # print(classifier_name)
    classifiers = list()

    for name in classifier_name:
        document = open("saved/" + name + ".p", "rb")
        classifier = pickle.load(document)
        classifiers.append(classifier)
        document.close()

    return classifiers


def LoadFeatures():
    documents_f = open("saved/word_features5k.p", "rb")
    word_features = pickle.load(documents_f)
    documents_f.close()
    return word_features