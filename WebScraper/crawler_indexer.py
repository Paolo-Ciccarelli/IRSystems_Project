#pip install nltk
#pip install lxml

import nltk,os,glob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from typing import Dict, List, Tuple

nltk.download('punkt') 
nltk.download('punkt_tab')   
nltk.download('stopwords')

TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+(?:'[\w]+)?")
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Transforms tokens into terms using linguistic preprocessing
def preprocess_tokenize(text: str) -> List[str]:
    #Case Folding
    text = text.lower()
    #Tokenization
    tokens = TOKENIZER.tokenize(text)
    #Performs stemming
    #Removes stopwords and very short tokens
    terms = []
    for token in tokens:
        if token not in STOPWORDS and len(token) >= 3:
            if token.isdigit():
                continue
            stemmed_term = STEMMER.stem(token)
            terms.append(stemmed_term)
    return terms

# Builds and maintains the inverted index incrementally throughout crawling
# Returns the full term list for that PDF for later use in clustering
def update_index(inverted_index: Dict[str, List[int]], doc_id: int, text: str) -> List[str]:
    terms = preprocess_tokenize(text)
    seen = set() # avoids duplicate docIDs for the same term
    for term in terms:
        # If the term has already been encountered in the PDF, skip
        if term in seen:
            continue
        seen.add(term)
        postings = inverted_index.setdefault(term, [])
        postings.append(doc_id)
    return terms

# Constructs the inverted index
# Hash Table (key: the term, value: its postings list)
def build_inverted_index(pairs: List[Tuple[str,int]]) -> Dict[str, List[int]]:
    index = {} 
    for term, docid in pairs:
        if term not in index:
            index[term] = []
        index[term].append(docid)
    print(f"DEBUG: Index contains {len(index)} unique terms")
    return index
