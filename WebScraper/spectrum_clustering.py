# pip install numpy
# pip install scikit-learn

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import query_processor

# Collecting all project component paths
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
INDEX_PATH = OUTPUT_DIR / "inverted_index.json"
TOKENS_PATH = OUTPUT_DIR / "document_tokens.json"
METADAT_PATH = OUTPUT_DIR / "pdf_metadata.json"

K_VALUES = [2,10,20]
print("HELLO")

# Loads the inverted index from the JSON file
def load_inverted_index() -> Dict[str, List[int]]:
    if not INDEX_PATH.exists(): 
        raise FileNotFoundError(f"ERROR: inverted index could not be found at {INDEX_PATH}. Ensure the crawler has been run first.")
    return query_processor.load_json_index(str(INDEX_PATH))

# Loads the list of tokens associated with each PDF from the JSON file
# Preapres each token list for use in the TfidfVectorizer module
def load_document_tokens() -> List[str]:
    if not TOKENS_PATH.exists():
        raise FileNotFoundError(f"ERROR: document_tokens.json could not be found at {TOKENS_PATH}. Ensure the crawler has been run first.")
    
    with open(TOKENS_PATH, "r", encoding="utf-8") as f:
        docs_tokens: List[List[str]] = json.load(f)
    doc_texts: List[str] = [" ".join(tokens) for tokens in docs_tokens]
    return doc_texts

def modified_lookup_singleQ(index: Dict[str, List[int]], term: str) -> List[int]:
    docIDs, elapsed = query_processor.lookup_singleQ(index, term)
    return sorted(set(docIDs))

# Constructs a new collection 'My_collection' that contains all documents from both queries without duplicates
# Returns:
#   sustainability_IDs: sorted unique doc IDs for 'sustainability'   
#   waste_IDs: sorted unique doc IDs for 'waste'
#   My_collection: sorted unique doc IDs for union(sustainability_IDs, waste_IDs)
def build_mycollection(index: Dict[str, List[int]]) -> Tuple[List[int], List[int], List[int]]:
    sustainability_docIDs = set(modified_lookup_singleQ(index, "sustainability"))
    waste_docIDs = set(modified_lookup_singleQ(index, "waste"))
    intersection_docIDs = sorted(sustainability_docIDs & waste_docIDs)
    My_collection = sorted(sustainability_docIDs | waste_docIDs)
    
    print("\n=== My Collection Statistics ===")
    print(f"# docs containing 'Sustainability': {len(sustainability_docIDs)}")
    print(f"# docs containing 'Waste': {len(waste_docIDs)}")
    print(f"# docs containing both 'Sustainability' and 'Waste': {len(intersection_docIDs)}")
    print(f"# docs in total without duplicates: {len(My_collection)}")
    return sustainability_docIDs, waste_docIDs, My_collection
    
# Creates two different TF-IDF representations of the same set of documents
# Global IDF: the idf values computed over all crawled thesis PDFs
# Local IDF: the idf values computed over My_collection only
def build_tfidf_matrices(document_tokens: List[str], Mycollection_docIDs: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not Mycollection_docIDs:
        raise ValueError("ERROR: My_collection is empty so there is nothing to cluster.")
    
    # Extracts the token string from only those documents belonging to My_collection
    Mycollection_tokens = [document_tokens[docID] for docID in Mycollection_docIDs]
    
    # max_df=0.6 means that terms appearing in more than 60% of documents are excluded
    # min_df=2 means that only terms appearing in at least 2 documents are included
    # norm='l2' means that vectors are normalized using the L2 norm
    # sublinear_tf is basically log-scale term frequency
    vectorizer_global = TfidfVectorizer(max_df=0.6,min_df=2,norm="l2",sublinear_tf=True)
    
    # Trains the vectorizer on all PDF theses extracted by the crawler
    # Computes IDF values in this scope and transforms all documents into TF-IDF vectors
    vectorizer_global.fit(document_tokens)
    
    # Extracts the list of all unique terms that the global vectorizer identified
    term_names = vectorizer_global.get_feature_names_out()
    
    # A vocabulary dictionary mapping each term to its column index in the TF-IDF matrix
    vocab = {term: index for index, term in enumerate(term_names)}
    
    # Applies the fitted global vectorizer to transform only documents belonging to My_collection
    # Computes IDF values from the entire corpus, creates TF-IDF vectors for My_collection documents
    X_global = vectorizer_global.transform(Mycollection_tokens)
    
    # Creates a brand new vectorizer for the same vocabulary as the global one
    vectorizer_local = TfidfVectorizer(norm="l2", sublinear_tf=True, vocabulary=vocab)
    
    # Trains the vectorizer on only the PDF theses belonging to My_collection
    # Recomputes IDF values based only on My_collection and transforms those documents
    X_local = vectorizer_local.fit_transform(Mycollection_tokens)
    
    return X_global, X_local, term_names


# X: the TF-IDF matrix, with documents as rows and terms as columns
# term_names: array of terms corresponding to columns in X
# k_values: a sequence of K values to try, being [2,10,20]
def execute_kmeans(X, term_names: np.ndarray, k_values: Sequence[int], writer=None):
    def log(msg=""):
        if writer:
            writer.write(msg+"\n")
        else:
            print(msg)
    
    # Loops through each of the K values, where K is the number of clusters
    for k in k_values:
        log(f"\n--- Cluster Run when K = {k} ---")
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=0)
        km.fit(X) # trains the KMeans model on the TF-IDF matrix T
        labels = km.labels_ #each entry is the cluster ID assigned to a given document
        centers = km.cluster_centers_ #2D array (k x #terms) where each row is a cluster centroid
        
        #2D array where each row contains feature indices ordered by their weight
        #[:,::-1] reverses order from ascending to descending sort
        order_centroids = np.argsort(centers, axis=1)[:,::-1] 
        
        # Loops each cluster for to print diagnostic information
        for cluster_index in range(k):
            cluster_size = int(np.sum(labels == cluster_index))
            log(f"\nCluster {cluster_index} (size = {cluster_size})")
            top_indices = order_centroids[cluster_index, :50] #top 50 vocab terms
            
            # Loops over each of the top 50 vocab terms
            for term_index in top_indices:
                term = term_names[term_index]
                weight = centers[cluster_index, term_index]
                log(f" {term:25s} {weight:4f}")

inverted_index = load_inverted_index()
document_tokens = load_document_tokens()
sustainability_docIDs, waste_docIDs, My_collection = build_mycollection(inverted_index)
X_global, X_local, term_names = build_tfidf_matrices(document_tokens, My_collection)


output_path = OUTPUT_DIR / "clustering_results.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== KMEANS RESULTS USING GLOBAL TF-IDF ===\n")
    execute_kmeans(X_global, term_names, K_VALUES, writer=f)
    
    f.write("\n\n=== KMEANS RESULTS USING LOCAL TF-IDF ===\n")
    execute_kmeans(X_local, term_names, K_VALUES, writer=f)
    
    