#Tagging and chunking

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

#Just obtaining texts
train_text=state_union.raw("2005-GWBush.txt") 
sample_text=state_union.raw("2006-GWBush.txt")

#Separating text by sentences
custom_sent_tokenizer=PunktSentenceTokenizer(sample_text)
tokenized=custom_sent_tokenizer.tokenize(sample_text)

#Recorre cada oración
for i in tokenized:
    words=nltk.word_tokenize(i)#Tokenize by word
    tagged=nltk.pos_tag(words)#Tag each word ***(Funciona para español???)***

    #Manually Chunking           
    #chunkGram=r"""Chunk: {<.*>+}#Chunk everything 
    #            }<VB.?|IN|DT>+{"""#Chink (NOT chunk)
    #chunkParser=nltk.RegexpParser(chunkGram)
    #chunked=chunkParser.parse(tagged)
    
    #NLTK name entity chunking
    chunked=nltk.ne_chunk(tagged,binary=True)
    print(chunked)
    chunked.draw()

"""
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
"""