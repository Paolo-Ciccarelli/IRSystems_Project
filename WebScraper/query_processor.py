import time
import json
from pathlib import Path
from nltk.stem import PorterStemmer
from typing import Dict, List, Tuple

STEMMER = PorterStemmer()

# Performs query normalization
def _normalize(term: str) -> str:
    return STEMMER.stem(term.lower())

# Implements Figure 1.6 from the textbook
# Evaluates the intersection of two postings lists p1 and p2
def intersect_postings(p1, p2):
    answer = []
    i, j = 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i += 1 
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return answer

# Processes a single term query
def lookup_singleQ(index: Dict[str, List[int]], term: str) -> List[int]:
    start_time = time.time()
    result = sorted(index.get(_normalize(term), []))
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

# Implements Figure 1.7 from the textbook
# Returns the set of documents containing each term in the input list of terms
def lookup_andQ(index: Dict[str, List[int]], *terms: str) -> List[int]:
    # dictionary: the naive inverted index
    # *terms: a tuple collecting all arguments after dictionary <t1,...,tn>
    if not terms: 
        return [], 0.0
    start_time = time.time()
    # Retrieves postings lists for all terms in *terms
    term_postings = []
    for t in terms:
        postings_list = sorted(index.get(_normalize(t), []))
        # Handles scenario where one or more terms have no postings
        if not postings_list:
            return []
        term_postings.append(postings_list)
    # Sorts shortest postings first for efficiency
    term_postings.sort(key=len) 
    intersect_result = term_postings[0]
    # Iteratively intersects remaining postings lists
    for next_postings in term_postings[1:]:
        # Handles scenario where intersection is already empty
        if not intersect_result:
            break
        intersect_result = intersect_postings(intersect_result, next_postings)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return intersect_result, elapsed_time

# Loads the inverted index built in the spectrum_spider.py module
# Hash Table (key: the unique term, value: its postings list)
def load_json_index(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)    
    return {term: list(map(int, postings)) for term, postings in data.items()}

if __name__ == "__main__":
    index_path = Path("output") / "inverted_index.json"
    if not index_path.exists():
        print(f"ERROR: inverted index at {index_path} could not be found. Be sure to run the crawler first.")
    else:
        inverted_index = load_json_index("C:\\Users\\User\\Downloads\\COMP479_P2_40286203\\output\\inverted_index.json")
        for term in ["sustainability", "waste"]:
            term_collection, query_time = lookup_singleQ(inverted_index, term)
            print(f"DEBUG: query lookup for '{term}': {len(term_collection)} documents, {query_time:.6f} seconds.")
            print(f"DEBUG: {term} found in: {term_collection}")
        my_collection, query_time = lookup_andQ(inverted_index, "sustainability", "waste")
        print(f"DEBUG: query lookup for 'sustainability' and 'waste': {len(my_collection)} documents, {query_time:.6f} seconds.")
        print(f"DEBUG: sustainability and waste found in: {my_collection}")
        
            