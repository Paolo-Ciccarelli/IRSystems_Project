#pip install nltk
#pip install lxml

import nltk, os, glob, time
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

# Reused from other modules
from naive_indexer import parse_sgm
from naive_indexer import preprocess_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+(?:'[\w]+)?")
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Memory limit for each block (measured in number of postings)
BLOCK_SIZE_LIMIT = 50000


# Implements SPIMI algorithm by building an in-memory inverted index from a stream of term-docID pairs, 
# until the block size limit is reached.
def spimi_invert(token_stream, block_num):    
    # A hash table mapping each term -> list of docIDs
    dictionary = defaultdict(list)
    postings_count = 0

    print(f"DEBUG: building block {block_num}...")
    for term, docid in token_stream:
        if not dictionary[term] or dictionary[term][-1] != docid:
            dictionary[term].append(docid)
            postings_count += 1
        if postings_count >= BLOCK_SIZE_LIMIT:
            break

    sorted_terms = sorted(dictionary.keys())
    block_filename = f'spimi_block_{block_num}.txt'
    write_to_disk(sorted_terms, dictionary, block_filename)
    print(f"DEBUG: block {block_num} written with {len(dictionary)} terms and {postings_count} postings.")
    return block_filename, postings_count

# Writes the block to disk in human-readable format, one term per line.
def write_to_disk(sorted_terms, dictionary, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for term in sorted_terms:
            postings = dictionary[term]
            f.write(f"{term}: {' '.join(map(str, postings))}\n")

# Combines multiple sorted block files into a single inverted index by executing the k-way merge 
# algorithm, also removes duplicates. 
def merge_blocks(block_files, output_file='spimi_inverted_index.txt'):
    return 0