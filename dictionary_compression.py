import naive_indexer
import nltk, os, glob
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('punkt-tab', quiet=True)
nltk.download('stopwords', quiet=True)

TOKENIZER = RegexpTokenizer(r"[A-Za-z0-9]+(?:'[\w]+)?")
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# =================================
# PREPROCESSING FUNCTIONS FOR TABLE
# =================================

# Basic tokenization without any filtering whatsoever (row I)
def tokenize_only(text):
    return TOKENIZER.tokenize(text)

# Removes all numbers from the token list (row II)
def remove_numbers(tokens):
    return [token for token in tokens if not token.isdigit()]

# Converts all tokens to lowercase equivalent (row III)
def case_fold(tokens):
    return [token.lower() for token in tokens]

# Removes all stopwords from the token list (row IV/V)
def remove_stopwords(tokens, stopword_list):
    return [token for token in tokens if token not in stopword_list]

# Reduces tokens to their root or base form (row VI)
def stem_tokens(tokens):
    return [STEMMER.stem(token) for token in tokens]

# ===================================
# HELPER FUNCTIONS FOR PRINTING TABLE
# ===================================

# Evaluates the percentage change from the previous value (Δ%)
# By convention: positive return for increase, negative return for decrease
def calculate_delta(current: int, previous: int):
    # exception handling to avoid division by zero
    if not previous:
        return 0
    difference = previous - current
    ratio = difference/previous
    percentage = -100 * ratio
    return round(percentage)

# Evaluates the cumulative percentage change from the baseline
def calculate_total(current, baseline):
    # exception handling to avoid division by zero
    if not baseline:
        return 0
    difference = baseline - current
    ratio = difference/baseline
    percentage = -100 * ratio
    return round(percentage)

# ===================================
# IMPORTANT FUNCTIONS
# ===================================

# Builds Table 5.1 across all preprocessing stages
# Each subsequent row is a cumulative reduction from the unfiltered set of tokens
def build_compression_table(directory):
    sgm_files = sorted(glob.glob(os.path.join(directory, '*sgm')))
    
    # Prepares the groundwork for building the table by generating custom stopword lists
    global_pairs = [] #stores (docID, text) tuples for every document in the corpus
    global_lctokens = [] #stores tokens that have been processed until casefolding

    for filepath in sgm_files:
        documents = naive_indexer.parse_sgm(filepath)
        global_pairs.extend(documents)
        for (docID, text) in documents:
            tokens = tokenize_only(text)
            tokens = remove_numbers(tokens)
            tokens_lower = case_fold(tokens)
            global_lctokens.extend(tokens_lower) 

    print(f"DEBUG: loaded {len(global_pairs)} documents")
    word_freq = Counter(global_lctokens)
    top_30 = set([word for word, count in word_freq.most_common(30)])
    top_150 = set([word for word, count in word_freq.most_common(150)])
    print(f"DEBUG: top 30 words: {list(top_30)[:10]}...")

    # Computes statistics for each preprocessing level
    preprocessing_stages = [
        ("unfiltered", lambda tokens: tokens),
        ("no_numbers", lambda tokens: remove_numbers(tokens)),
        ("case_folding", lambda tokens: case_fold(remove_numbers(tokens))),
        ("stop_30", lambda tokens: remove_stopwords(case_fold(remove_numbers(tokens)), top_30)),
        ("stop_150", lambda tokens: remove_stopwords(case_fold(remove_numbers(tokens)), top_150)),
        ("stemming", lambda tokens: stem_tokens(remove_stopwords(case_fold(remove_numbers(tokens)), top_150)))
    ]
    
    # stages_F: dictionary storing (term,docID) pair lists for each stage
    stages_F = {stage_name: [] for stage_name, _ in preprocessing_stages}

    # Loops through all preprocessing stages for each document
    for docID, text in global_pairs:
        tokens = tokenize_only(text)
        for stage_name, preprocess_func in preprocessing_stages:
            terms = preprocess_func(tokens)
            stages_F[stage_name].extend((term, docID) for term in terms)

    # results: dictionary that holds the statistics for each column
    # First column holds the number of distinct terms after preprocessing
    # Second column holds the number of unique (term, docID) combinations
    results = {}
    for stage_name in stages_F:
        pairs = stages_F[stage_name]
        distinct_terms = len(set(term for term, docid in pairs))
        nonpos_postings = len(set(pairs))
        results[stage_name] = {
            'terms': distinct_terms,      # For column 1
            'postings': nonpos_postings,  # For column 2
        }
    return results



def print_table(results):
    stages_code = ['unfiltered', 'no_numbers', 'case_folding', 'stop_30', 'stop_150', 'stemming']
    stages_print = ['unfiltered', 'no numbers', 'case folding', '30 stop words', '150 stop words', 'stemming']
    
    print("\n"+"="*80)
    print("Dictionary Compression Table (5.1)")
    print("="*80)
    print(f"\n{'':20} {'(distinct) terms':>25} {'nonpositional postings':>25}")
    print(f"{'':20} {'number':>10} {'Δ%':>5} {'T%':>5}   {'number':>10} {'Δ%':>5} {'T%':>5}")
    print("-"*80)

    #Retrieve baseline (unfiltered) values
    baseline_terms = results['unfiltered']['terms']
    baseline_postings = results['unfiltered']['postings']

    prev_terms = baseline_terms
    prev_postings = baseline_postings
    for i, stage in enumerate(stages_code):
        current = results[stage]
        if stage in ['stop_30', 'stop_150']:
            prev_stage = results['case_folding']
        elif i > 0:
            prev_stage = results[stages_code[i-1]]
        else:
            prev_stage = None

        t_delta = calculate_delta(current['terms'], prev_stage['terms'] if prev_stage else None)
        p_delta = calculate_delta(current['postings'], prev_stage['postings'] if prev_stage else None)
        t_total = calculate_total(current['terms'], baseline_terms)
        p_total = calculate_total(current['postings'], baseline_postings)
        
        # Print row with formatted numbers and percentages
        print(f"{stages_print[i]:20} {current['terms']:10,} {abs(t_delta):5} {abs(t_total):5}   "
              f"{current['postings']:10,} {abs(p_delta):5} {abs(p_total):5}")

        # Update previous values for the next iteration
        if prev_stage:
            prev_terms = prev_stage['terms']
            prev_postings = prev_stage['postings']

    print("="*80)


# Testing 
reuters_dir = 'C:\\Users\\prowl\\Downloads\\reuters21578'   
results = build_compression_table(reuters_dir)
print_table(results)


