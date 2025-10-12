#pip install nltk
#pip install lxml

import nltk,os,glob
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

nltk.download('punkt') 
nltk.download('punkt_tab')   
nltk.download('stopwords')

TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+(?:'[\w]+)?")
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Extracts individual documents from a given .sgm file
def parse_sgm(filepath):
    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    documents = []
    for reuters_tag in soup.find_all('reuters'):
        newid = int(reuters_tag.get('newid'))
        text_tag = reuters_tag.find('text')
        if not text_tag:
            continue

        title = text_tag.find('title')
        dateline = text_tag.find('dateline')
        body = text_tag.find('body')

        components = []
        if title: 
            components.append(title.get_text(" ", strip=True))
        if dateline: 
            components.append(dateline.get_text(" ", strip=True))
        if body:
            components.append(body.get_text(" ", strip=True))
        else:
            #Fallback option in case there's no body
            components.append(text_tag.get_text(" ", strip=True))
        raw_text = " ".join(components)
        documents.append((newid, raw_text))
    print(f"DEBUG: {len(documents)} documents parsed.")
    return documents

# Transforms tokens into terms using linguistic preprocessing
def preprocess_tokenize(text):
    #Lowercase normalization
    text = text.lower()
    #Tokenization
    tokens = TOKENIZER.tokenize(text)

    #Performs stemming
    #Removes stopwords and very short tokens
    terms = []
    for token in tokens:
        if token not in STOPWORDS and len(token) >= 2:
            stemmed_term = STEMMER.stem(token)
            terms.append(stemmed_term)
    return terms

#Processes ALL documents and accumulate term-docID pairs in list F
def process_documents(directory):
    F = []
    total_docs = 0
    print("DEBUG: Building term-docID pairs...")

    sgm_files = sorted(glob.glob(os.path.join(directory, '*.sgm')))
    for filepath in sgm_files:
        filename = os.path.basename(filepath)
        print(f"\nDEBUG: Processing {filename}...")
        documents = parse_sgm(filepath)
        for docid, text in documents:
            terms = preprocess_tokenize(text)
            F.extend((term,docid) for term in terms)
        total_docs += len(documents)

    print(f"DEBUG: Processed {total_docs} documents from {len(sgm_files)} files")
    print(f"DEBUG: Generated {len(F)} term-docID pairs")
    return F

#Sorts F alphabetically and removes duplicates
def sort_cull(F):
    print("DEBUG: Sorting and removing duplicates...")
    #Sort by term first, then by docID
    F_sorted = sorted(F)
    print(f"DEBUG: Sorted {len(F_sorted)} pairs")

    F_unique = []
    prev_pair = None
    for pair in F_sorted:
        if pair != prev_pair:
            F_unique.append(pair)
        prev_pair = pair
    
    duplicates_removed = len(F_sorted) - len(F_unique)
    print(f"DEBUG: Removed {duplicates_removed} duplicates")
    print(f"DEBUG: Unique pairs: {len(F_unique)}")
    return F_unique

# Constructs the inverted index
# Hash Table (key: the term, value: its postings list)
def build_inverted_index(F_sorted):
    index = {} 
    for term, docid in F_sorted:
        if term not in index:
            index[term] = []
        index[term].append(docid)
    print(f"DEBUG: Index contains {len(index)} unique terms")
    return index

# Display index statistics and sample
def display_index_info(index):
    print("\n" + "="*60)
    print("INVERTED INDEX STATISTICS")
    print("="*60)
    
    print(f"\nTotal unique terms: {len(index)}")
    
    # Calculate statistics
    postings_lengths = [len(postings) for postings in index.values()]
    total_postings = sum(postings_lengths)
    avg_postings = total_postings / len(index) if index else 0
    max_postings = max(postings_lengths) if postings_lengths else 0
    
    print(f"Total postings: {total_postings}")
    print(f"Average postings per term: {avg_postings:.2f}")
    print(f"Maximum postings for a term: {max_postings}")
    
    # Find most common terms
    print("\n" + "-"*60)
    print("TOP 20 MOST FREQUENT TERMS:")
    print("-"*60)
    
    sorted_terms = sorted(index.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (term, postings) in enumerate(sorted_terms[:20], 1):
        print(f"{i:2d}. '{term}' appears in {len(postings)} documents: {postings}")
    
    # Sample some terms alphabetically
    print("\n" + "-"*60)
    print("SAMPLE TERMS (Alphabetically):")
    print("-"*60)
    
    sample_terms = sorted(index.keys())[:20]
    for term in sample_terms:
        postings = index[term]
        print(f"'{term}' -> {postings}")




print("="*60)
print("NAIVE INDEXER IMPLEMENTATION")
print("="*60)
reuters_dir = 'C:\\Users\\prowl\\Downloads\\reuters21578'    
F = process_documents(reuters_dir)
F_sorted = sort_cull(F)
inverted_index = build_inverted_index(F_sorted)
#display_index_info(inverted_index)
