# python -m pip install nltk
# python -m pip install numpy pandas

import pandas as pd
from nltk.corpus import reuters
import nltk
nltk.download('reuters')
nltk.download('averaged_perceptron_tagger_eng')
print("DEBUG: reuters corpus downloaded")
print("DEBUG: averaged_perceptron_tagger_eng downloaded")

# Task (iii)
num_documents = len(reuters.fileids())
num_words = len(reuters.words())
corpus_sentences = reuters.sents()
num_sentences = len(corpus_sentences)
print("Number of documents: ", num_documents) #should be 10788
print("Number of words: ", num_words) #should be 1720901
print("Number of sentences: ", num_sentences) #should be 54716

# Task (iv)
num_words_9920 = reuters.words('training/9920')
def count_prepositions(fileID):
    words = reuters.words(fileID) #fetches all words in document
    pos_tags = nltk.pos_tag(words) #assigns POS tag to each word
    prepositions = []
    for word, pos in pos_tags:
        if pos == 'IN': #IN means preposition
            prepositions.append(word)
    return len(prepositions)
num_prepositions_9920 = count_prepositions('training/9920')
print("Number of words in 9920: ", len(num_words_9920))
print("Number of prepositions in 9920: ", num_prepositions_9920)

# Task (v)
print("Number of categories: ", len(reuters.categories()))
rows = reuters.categories()
dataset = {}
for row in rows:
    dataset[row] = reuters.fileids(row)
df = pd.DataFrame(dataset)
print(df)

# Task (vi)
def word_freq(target_word, fileID):
    words = reuters.words(fileID)
    frequency = 0
    for word in words:
        if word == target_word:
            frequency += 1
    return frequency
print("How many times does 'of' appear in 9920?: ", word_freq('of','training/9920'))
print(reuters.raw('training/9920')) #there are indeed 2 instances of the word 'of'




