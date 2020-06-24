#Text classification
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

#List of tupples (review,category (positive or negative rating))
documents=[]
positive=open("positive.txt","r").read()
negative=open("negative.txt","r").read()

for review in positive.split('\n'):
    documents.append((word_tokenize(review),"pos"))
for review in negative.split('\n'):
    documents.append((word_tokenize(review),"neg"))

#Shuffle order of movies
random.shuffle(documents)
print(documents[0][0])

#Grap ALL the words of ALL the reviews together
allowed_types=["J"]#Adjectives
all_words=[]
positive_words_tagged=nltk.pos_tag(word_tokenize(positive))
negative_words_tagged=nltk.pos_tag(word_tokenize(negative))

for w in positive_words_tagged:
    if w[1][0] in allowed_types:
        all_words.append(w[0].lower())

for w in negative_words_tagged:
    if w[1][0] in allowed_types:
        all_words.append(w[0].lower())

all_words=nltk.FreqDist(all_words)#Make a frequency distribution of all words to analyse

print(all_words.most_common(10))#Most used words
#print(all_words["stupid"])#Times stupid appears

top_words=list(all_words.keys())[:1300]#List of top 3000 words
print(type(top_words))
print(len(top_words))

save_pickle=open("documents.pickle","wb")
pickle.dump(documents,save_pickle)
save_pickle.close()

save_pickle=open("features.pickle","wb")
pickle.dump(top_words,save_pickle)
save_pickle.close()


#Functions wit dictionary {topword, true or false in text given}
def find_inTopWords(documentwords):
    words=set(documentwords)
    #print(type(documentwords))
    features={}
    for w in top_words:
        features[w]=(w in words)

    return features

#List of features of all reviews
featuresets=[]
for (review,rating) in documents:
    featureset=find_inTopWords(review)
    featuresets.append((featureset,rating))

training_set=featuresets[:10000]
testing_set=featuresets[10000:]

#*************************************MODELS******************************************************************
#Naive Bayes Classifier algorithms
#Nltk default NB
classifier=nltk.NaiveBayesClassifier.train(training_set)
accuracy=(nltk.classify.accuracy(classifier,testing_set))*100
print("Original Accuracy: ",accuracy)
classifier.show_most_informative_features(15)

#Multinomial NB
MN_classifier=SklearnClassifier(MultinomialNB())
MN_classifier.train(training_set)
accuracy=(nltk.classify.accuracy(MN_classifier,testing_set))*100
print("Multinomial class Accuracy: ",accuracy)

#Bernoulli NB
B_classifier=SklearnClassifier(BernoulliNB())
B_classifier.train(training_set)
accuracy=(nltk.classify.accuracy(B_classifier,testing_set))*100
print("Bernoulli class Accuracy: ",accuracy)

#Linear Model

#Logistic Regression
LogisticRegression_classifier=SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
accuracy=(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100
print("LogisticRegression class Accuracy: ",accuracy)

#SGDClassifier
SGDClassifier_classifier=SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
accuracy=(nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100
print("SGDClassifier class Accuracy: ",accuracy)

#svm

#SVC
SVC_classifier=SklearnClassifier(SVC())
SVC_classifier.train(training_set)
accuracy=(nltk.classify.accuracy(SVC_classifier,testing_set))*100
print("SVC class Accuracy: ",accuracy)

#LinearSVC
#LinearSVC_classifier=SklearnClassifier(LinearSVC())
#LinearSVC_classifier.train(training_set)
#accuracy=(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100
#print("LinearSVC class Accuracy: ",accuracy)

#NuSVC
#NuSVC_classifier=SklearnClassifier(NuSVC())
#NuSVC_classifier.train(training_set)
#accuracy=(nltk.classify.accuracy(NuSVC_classifier,testing_set))*100
#print("NuSVC class Accuracy: ",accuracy)

#Vote Classifier
vote_classifier=VoteClassifier(
                                classifier,
                                MN_classifier,
                                B_classifier,
                                LogisticRegression_classifier,
                                SGDClassifier_classifier,
                                SVC_classifier,
                                #LinearSVC_classifier,
                                #NuSVC_classifier
                            )
accuracy=(nltk.classify.accuracy(vote_classifier,testing_set))*100
print("Vote class Accuracy: ",accuracy)

#Saving and opening model
#Saving

save_classifier=open("NaiveBayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

save_classifier=open("MN_classifier.pickle","wb")
pickle.dump(MN_classifier,save_classifier)
save_classifier.close()

save_classifier=open("B_classifier.pickle","wb")
pickle.dump(B_classifier,save_classifier)
save_classifier.close()

save_classifier=open("LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_classifier)
save_classifier.close()

save_classifier=open("SGDClassifier_classifier.pickle","wb")
pickle.dump(SGDClassifier_classifier,save_classifier)
save_classifier.close()

save_classifier=open("SVC_classifier.pickle","wb")
pickle.dump(SVC_classifier,save_classifier)
save_classifier.close()

#save_classifier=open("LinearSVC_classifier.pickle","wb")
#pickle.dump(LinearSVC_classifier,save_classifier)
#save_classifier.close()

#save_classifier=open("NuSVC_classifier.pickle","wb")
#pickle.dump(NuSVC_classifier,save_classifier)
#save_classifier.close()


#Opening
#classifier_f=open("NaivesBayes.pickle","rb")
#classifier=pickle.load(classifier_f)
#classifier_f.close()