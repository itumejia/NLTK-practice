#Lemmatizing and wordnet

from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

#Find practical meanings for words. Spanish?
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better","a"))#Find adjective for that word

#Wordnet. Spanish?
from nltk.corpus import wordnet
syns=wordnet.synsets("program")#Set of synonyms for program
print(syns)#Print set
print(syns[0].lemmas()[0].name())#Print word
print(syns[0].definition())#definition
print(syns[0].examples())#examples

#Synonyms and antonyms of good

synonyms=[]
anonyms=[]

for syn in wordnet.synsets("good"):
    for lemma in syn.lemmas():
        print(lemma)
        synonyms.append(lemma.name())
        if lemma.antonyms():
            anonyms.append(lemma.antonyms()[0].name())

print((synonyms))
print((anonyms))


#Semantic similarity
w1=wordnet.synset("ship.n.01")
w2=wordnet.synset("desktop.n.01")
print(type(w1))
print(w1.wup_similarity(w2))

