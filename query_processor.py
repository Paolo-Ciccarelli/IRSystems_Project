import time
from nltk.stem import PorterStemmer
from typing import Dict, List, Iterable

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


from naive_indexer import inverted_index
# Single Term test queries
print("Searching up 'lawsuit':", lookup_singleQ(inverted_index,"lawsuit")) 
print("Seaching up 'bankruptcy':", lookup_singleQ(inverted_index,"bankruptcy"))
print("Seaching up 'hollywood':",lookup_singleQ(inverted_index,"hollywood"))

# Multiple Term test queries
print("\nSearching up 'liberal' and 'conservative':", lookup_andQ(inverted_index, "liberal", "conservative")) 
print("\nSeaching up 'supreme' and 'court':", lookup_andQ(inverted_index, "supreme", "court"))
print("\nSeaching up 'cold' and 'war':",lookup_andQ(inverted_index, "cold", "war"))

# Challenge queries
print("\nChallenge Query #1 'copper':", lookup_singleQ(inverted_index,"copper"))
print("\nChallenge Query #2 'Chrysler':", lookup_singleQ(inverted_index,"Chrysler"))
print("\nChallenge Query #3 'Bundesbank':", lookup_singleQ(inverted_index,"Bundesbank"))