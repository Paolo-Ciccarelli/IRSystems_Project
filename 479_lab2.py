# pip install nltk

from nltk.corpus import reuters
import nltk
nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')

print("DEBUG: reuters corpus downloaded")
num_documents = len(reuters.fileids())
num_words = len(reuters.words())

corpus_sentences = []
for fileid in reuters.fileids():
    file_sentences = nltk.sent_tokenize(reuters.raw(fileid))
    corpus_sentences.extend(file_sentences)
num_sentences = len(corpus_sentences)

print("Number of documents: ", num_documents) #should be 10788
print("Number of words: ", num_words) #should be
print("Number of sentences: ", num_sentences)