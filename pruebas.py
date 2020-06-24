#Tokenizing,stopwords, stemming

from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer #Cambio para flow: Stemmer en inglés, buscar para españool

example_text= "Hello there Sr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat carboard. run running runned correr corriendo corrio correra"

sentences=(sent_tokenize(example_text))#Separa por oraciones
words=(word_tokenize(example_text))#Separa por palabra

#for word in word_tokenize(example_text):
#   print(word)

stop_words=set(stopwords.words("english")) #Lista de palabras que no sirven. Cambio para flow: spanish
print(stop_words)

filtered=[]

ps=PorterStemmer()

#Eliminar stop words del texto
for word in words:
    if word.lower() not in stop_words:
        filtered.append(ps.stem(word))#Quedarse solo con la base de las palabras

print(filtered)