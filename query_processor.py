from nltk.stem import PorterStemmer
STEMMER = PorterStemmer()

# Processes a single term query
def lookup_singleQ(dictionary, term):
    term = STEMMER.stem(term.lower())
    return dictionary.get(term, [])

# Implements Figure 1.6 from the textbook
# Evaluates the intersection of two postings lists p1 and p2
def intersect_postings(p1, p2):
    answer = []
    i = 0, j = 0
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

# Implements Figure 1.7 from the textbook
# Returns the set of documents containing each term in the input list of terms
def lookup_andQ(dictionary, *terms):
    # dictionary: the naive inverted index
    # *terms: a tuple collecting all arguments after dictionary <t1,...,tn>
    if not terms: 
        return []
    normalized_terms = [STEMMER.stem(term.lower()) for term in terms]
    # Retrieves postings lists for all terms in *terms
    postings_lists = []
    for term in normalized_terms:
        postings = dictionary.get(term, [])
        # Handles scenario where any term has no postings
        if not postings:
            return []
        postings_lists.append((term, postings))
    postings_lists.sort(key=lambda x: len(x[1])) #sorts by increasing size
    result = postings_lists[0][1]
    for i in range (1, len(postings_lists)):
        if not result:
            break
        result = intersect_postings(result, postings_lists[i][1])
    return result


