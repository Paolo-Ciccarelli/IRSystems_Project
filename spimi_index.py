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
    print(f"\nDEBUG: merging {len(block_files)} blocks...")

    file_handles = []
    block_lines = []

    for block_file in block_files:
        fh = open(block_file, 'r', encoding='utf-8')
        file_handles.append(fh)
        line = fh.readline()
        if line:
            term, postings_str = line.strip().split(':', 1)
            postings = list(map(int, postings_str.split()))
            block_lines.append((term, postings, len(file_handles)-1))

    merged_index = defaultdict(list)
    while block_lines:
        # Find the minimum term
        block_lines.sort(key=lambda x: x[0])
        min_term = block_lines[0][0]

        # Collect all postings for this term from all blocks
        postings_for_term = []
        indices_to_remove = []
        for i, (term, postings, file_idx) in enumerate(block_lines):
            if term == min_term:
                postings_for_term.extend(postings)
                indices_to_remove.append(i)
        # Remove processed entries and read next lines
        for i in reversed(indices_to_remove):
            file_idx = block_lines[i][2]
            block_lines.pop(i)

            # Read next line from this file
            line = file_handles[file_idx].readline()
            if line:
                term, postings_str = line.strip().split(':', 1)
                postings = list(map(int, postings_str.split()))
                block_lines.append((term, postings, file_idx))
        
        # Merge and sort postings + remove duplicates
        merged_index[min_term] = sorted(set(postings_for_term))
    
    # Closes all file handles for maintenance
    for fh in file_handles:
        fh.close()

    # Writes the merged inverted index to disk in human-readable format
    with open(output_file, 'w', encoding='utf-8') as f:
        for term in sorted(merged_index.keys()):
            postings = merged_index[term]
            f.write(f"{term}: {' '.join(map(str, postings))}\n")
    
    # Cleans up the block files
    for block_file in block_files:
        if os.path.exists(block_file):
            os.remove(block_file)
    return merged_index

def build_spimi_index(directory, max_docs=None):
    return 0