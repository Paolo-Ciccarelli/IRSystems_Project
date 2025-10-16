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

# Invoked during the cumulative preprocessing stages
def chain_base(tokens):
    return case_fold(remove_numbers(tokens))

# ===================================
# HELPER FUNCTIONS FOR PRINTING TABLE
# ===================================

# Evaluates the percentage change from the previous value (Δ%)
# By convention: positive return for increase, negative return for decrease
def calculate_delta(current: int, previous: int):
    # exception handling to protect against division by zero
    if not previous:
        return 0
    difference = previous - current
    ratio = difference/previous
    return round(-100 * ratio)

# Evaluates the cumulative percentage change from the baseline
def calculate_total(current, baseline):
    # exception handling to protect against division by zero
    if not baseline:
        return 0
    difference = baseline - current
    ratio = difference/baseline
    return round(-100 * ratio)

# ===================================
# IMPORTANT FUNCTIONS FOR PART I
# ===================================

# Builds Table 5.1 across all preprocessing stages
# Each subsequent row is a cumulative reduction from the unfiltered set of tokens
def build_compression_table(directory):
    sgm_files = sorted(glob.glob(os.path.join(directory, '*sgm')))
    
    # First Pass: scans corpus to identify the 30/150 most common stopwords
    token_counts = Counter()
    for filepath in sgm_files:
        documents = naive_indexer.parse_sgm(filepath)
        for _, text in documents:
            tokens = tokenize_only(text)
            lctokens = case_fold(remove_numbers(tokens))
            token_counts.update(lctokens)
    
    # Generates custom stopword lists
    top_30_stops = {word for word, _ in token_counts.most_common(30)}
    top_150_stops = {word for word, _ in token_counts.most_common(150)}
    
    # Defines the preprocessing stages (ie: the rows of the table)
    preprocessing_stages = {
        'unfiltered': lambda t: t,
        'no_numbers': remove_numbers,
        'case_folding': chain_base,
        'stop_30': lambda t: remove_stopwords(chain_base(t), top_30_stops),
        'stop_150': lambda t: remove_stopwords(chain_base(t), top_150_stops),
        'stemming': lambda t: stem_tokens(remove_stopwords(chain_base(t), top_150_stops))
    }
    
    # Second Pass: re-scans corpus to apply filters and count savings
    # stages_F: dictionary storing refined (term,docID) pairs for each preprocessing stage
    stages_F = {stage: [] for stage in preprocessing_stages}
    for filepath in sgm_files:
        documents = naive_indexer.parse_sgm(filepath)
        for docID, text in documents:
            tokens = tokenize_only(text)
            for stage_name, preprocess_func in preprocessing_stages.items():
                terms = preprocess_func(tokens)
                stages_F[stage_name].extend((term, docID) for term in terms)

    # results: aggregates raw data from "stages_F" into counts for future analysis
    # First column holds the number of distinct terms after preprocessing
    # Second column holds the number of unique (term, docID) combinations
    results = {}
    for stage_name in stages_F:
        pairs = stages_F[stage_name]
        distinct_terms = len(set(term for term, _ in pairs))
        nonpos_postings = len(set(pairs))
        results[stage_name] = {
            'terms': distinct_terms,      # For column 1
            'postings': nonpos_postings,  # For column 2
        }
    return results, stages_F, top_30_stops, top_150_stops

# Prints the final compression table to the console
# Evaluates Δ% and T% respectively during construction
def print_table(results):
    stages_code = ['unfiltered', 'no_numbers', 'case_folding', 'stop_30', 'stop_150', 'stemming']
    stages_print = ['unfiltered', 'no numbers', 'case folding', '30 stop words', '150 stop words', 'stemming']
    
    print("\n"+"="*80)
    print("Dictionary Compression Table (5.1)")
    print("="*80)
    print(f"\n{'':20} {'(distinct) terms':>25} {'nonpositional postings':>25}")
    print(f"{'':20} {'number':>10} {'Δ%':>5} {'T%':>5}   {'number':>10} {'Δ%':>5} {'T%':>5}")
    print("-"*80)

    #Retrieves the baseline (unfiltered) values
    baseline_terms = results['unfiltered']['terms']
    baseline_postings = results['unfiltered']['postings']

    prev_terms = baseline_terms
    prev_postings = baseline_postings
    for i, stage in enumerate(stages_code):
        current = results[stage]
        if stage in ['stop_30', 'stop_150']:
            prev_terms_temp = results['case_folding']['terms']
            prev_postings_temp = results['case_folding']['postings']
        else:
            prev_terms_temp = prev_terms
            prev_postings_temp = prev_postings

        t_delta = calculate_delta(current['terms'], prev_terms_temp)
        p_delta = calculate_delta(current['postings'], prev_postings_temp)
        t_total = calculate_total(current['terms'], baseline_terms)
        p_total = calculate_total(current['postings'], baseline_postings)
        
        # Print row with formatted numbers and percentages
        print(f"{stages_print[i]:20} {current['terms']:10,} {t_delta:>5} {t_total:>5}   "
              f"{current['postings']:10,} {p_delta:>5} {p_total:>5}")

        # Update previous values for the next iteration
        prev_terms = current['terms']
        prev_postings = current['postings']

    print("="*80)


# Testing 
reuters_dir = 'C:\\Users\\prowl\\Downloads\\reuters21578'   
results, _, _, _ = build_compression_table(reuters_dir)
print_table(results)


