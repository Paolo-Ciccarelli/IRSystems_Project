#pip install nltk
#pip install lxml

import nltk, os, glob, time
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


# Builds inverted index by appending docIds to postings lists using a hash table.
# Showcases SPIMI's O(1) insertion with no O(T log T) sorting as with the naive indexer.
def build_spimi_inspired(directory):
    print("DEBUG: building SPIMI-inspired index.")
    total_docs = 0
    total_postings = 0
    inverted_index = defaultdict(list)
    
    start_time = time.time()
    sgm_files = sorted(glob.glob(os.path.join(directory, '*.sgm')))
    for filepath in sgm_files:
        documents = parse_sgm(filepath)
        for docid, text in documents:
            total_docs += 1
            terms = preprocess_tokenize(text)
            # where the SPIMI innovation kicks in
            # O(1) insertion per term, no sorting necessary
            for term in terms:
                if not inverted_index[term] or inverted_index[term][-1] != docid:
                    inverted_index[term].append(docid)
                    total_postings += 1
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Analysis
    print(f"\n=====SPIMI INVERTED INDEX RECAP=====")
    print(f"DEBUG: total documents processed is {total_docs}")
    print(f"DEBUG: total unique terms is {len(inverted_index)}")
    print(f"DEBUG: total postings is {total_postings}")
    print(f"DEBUG: process took {elapsed_time:.2f} seconds")
    print(f"DEBUG: average time per document is {elapsed_time/total_docs:.4f} seconds")

    # Writes SPIMI inverted index to file for inspection
    output_file = 'spimi_inverted_index.txt'
    print(f"\nDEBUG: writing inverted index to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for term in sorted(inverted_index.keys()):
            postings = inverted_index[term]
            f.write(f"{term}: {' '.join(map(str, postings))}\n")
    return inverted_index, elapsed_time, total_docs


def compare_with_naive(directory, num_pairs=10000):
    print("="*70)
    print("TIMING COMPARISON: SPIMI VS NAIVE")
    print("="*70)

    print(f"\nDEBUG: Collecting exactly {num_pairs} term-docID pairs from Reuters...")
    pairs = []
    sgm_files = sorted(glob.glob(os.path.join(directory, '*.sgm'))) 
    for filepath in sgm_files:
        documents = parse_sgm(filepath)
        for docid, text in documents:
            terms = preprocess_tokenize(text)
            for term in terms:
                pairs.append((term, docid))
                if len(pairs) >= num_pairs:
                    break
            if len(pairs) >= num_pairs:
                break
        if len(pairs) >= num_pairs:
            break
    pairs = pairs[:num_pairs]

    # =============================
    # SPIMI IMPLEMENTATION ANALYSIS
    # =============================
    spimi_start = time.time()
    spimi_index = defaultdict(list)
    for term, docid in pairs:
        if not spimi_index[term] or spimi_index[term][-1] != docid:
            spimi_index[term].append(docid)
    spimi_end = time.time()
    spimi_time = spimi_end - spimi_start
    print(f"\nDEBUG: SPIMI completed in {spimi_time:.6f} seconds")
    print(f"DEBUG: the number of unique terms indexed is {len(spimi_index)}\n")

    # =============================
    # NAIVE IMPLEMENTATION ANALYSIS
    # =============================
    naive_start = time.time()
    
    # process_documents() equivalent
    F = []
    for term, docid in pairs:
        F.append((term, docid))
    
    # sort_cull() equivalent 
    F_sorted = sorted(pairs)
    F_unique = []
    prev_pair = None
    for pair in F_sorted:
        if pair != prev_pair:
            F_unique.append(pair)
        prev_pair = pair
    
    # build_inverted_index() equivalent
    naive_index = {}
    for term, docid in F_unique:
        if term not in naive_index:
            naive_index[term] = []
        naive_index[term].append(docid)

    naive_end = time.time()
    naive_time = naive_end - naive_start
    print(f"DEBUG: NAIVE completed in {naive_time:.6f} seconds")
    print(f"DEBUG: the number of unique terms indexed is {len(naive_index)}\n")

    # =============================
    # COMPARATIVE ANALYSIS
    # =============================
    print(f"From theory, the SPIMI implementation is expected to outperform the Naive implementation.")
    print(f"Working with 10000 (term,docID) pairs, Naive took {naive_time:.6f} seconds and SPIMI took {spimi_time:.6f} seconds.")
    print(f"In other words, SPIMI outperformed Naive by {naive_time-spimi_time:.6f} seconds and was {(naive_time-spimi_time)/(spimi_time)*100:.2f}% faster.")

# Testing
reuters_dir = 'C:\\Users\\prowl\\Downloads\\reuters21578'  
compare_with_naive(reuters_dir)
