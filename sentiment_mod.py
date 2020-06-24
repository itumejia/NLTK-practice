import os
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
import pickle
from statistics import mode
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

def sentiment(text):
    feats=find_inTopWords(text)
    rating=vote_classifier.classify(feats)
    conf=vote_classifier.get_confidence()
    return rating,conf

class VoteClassifier(ClassifierI):
    conf=0
    def __init__(self,*classifiers):
        self._classifiers=classifiers

    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        choice=mode(votes)
        choice_votes=votes.count(choice)
        self.conf=choice_votes/len(votes)
        return mode(votes)
    def get_confidence(self):
        return self.conf

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pickle_open=open("documents.pickle","rb")
documents=pickle.load(pickle_open)
pickle_open.close()

pickle_open=open("features.pickle","rb")
top_words=pickle.load(pickle_open)
pickle_open.close()

def find_inTopWords(documentwords):
    documentwords=word_tokenize(documentwords)
    words=set(documentwords)
    #print(type(documentwords))
    features={}
    for w in top_words:
        features[w]=(w in words)

    return features

pickle_open=open("NaiveBayes.pickle","rb")
classifier=pickle.load(pickle_open)
pickle_open.close()

pickle_open=open("MN_classifier.pickle","rb")
MN_classifier=pickle.load(pickle_open)
pickle_open.close()

pickle_open=open("B_classifier.pickle","rb")
B_classifier=pickle.load(pickle_open)
pickle_open.close()

pickle_open=open("LogisticRegression_classifier.pickle","rb")
LogisticRegression_classifier=pickle.load(pickle_open)
pickle_open.close()

pickle_open=open("SGDClassifier_classifier.pickle","rb")
SGDClassifier_classifier=pickle.load(pickle_open)
pickle_open.close()

#pickle_open=open("SVC_classifier.pickle","rb")
#SVC_classifier=pickle.load(pickle_open)
#pickle_open.close()

vote_classifier=VoteClassifier(
                                classifier,
                                MN_classifier,
                                B_classifier,
                                LogisticRegression_classifier,
                                SGDClassifier_classifier,
                                #SVC_classifier,
                            )

