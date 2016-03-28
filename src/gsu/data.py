from nltk.tokenize import word_tokenize
from nltk import pos_tag, FreqDist
import random
import pickle


def DataSources():
    positiveData = open("resources/positive.txt", "r", encoding='utf-8', errors='replace').read()
    negativeData = open("resources/negative.txt", "r", encoding='utf-8', errors='replace').read()
    return positiveData, negativeData


def PrepareData():
    train_pos, train_neg = DataSources()
    documents = []
    all_words = []
    
#    j is adjective, r is adverb, and v is verb
#    allowed_word_types = ["J","R","V"]
    allowed_word_types = ["J"]

    for p in train_pos.split('\n'):
        documents.append((p, "pos"))
        words = word_tokenize(p)
        pos = pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    for p in train_neg.split('\n'):
        documents.append((p, "neg"))
        words = word_tokenize(p)
        pos = pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
    
    save_documents = open("saved/documents.p", "wb")
    pickle.dump(documents, save_documents)
    save_documents.close()

    all_words = FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]

    save_word_features = open("saved/word_features5k.p", "wb")
    pickle.dump(word_features, save_word_features)
    save_word_features.close()

    features = [(find_features(rev, word_features), category) for (rev, category) in documents]
    return features


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def TestTrainData():
    featuresets = PrepareData()
    random.shuffle(featuresets)
#    print(len(featuresets))
    testing_set = featuresets[10000:]
    training_set = featuresets[:10000]
    return training_set, testing_set