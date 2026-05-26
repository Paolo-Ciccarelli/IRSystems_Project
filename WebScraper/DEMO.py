# Initial Steps: Install these six Python packages using the terminal
# pip install scrapy
# pip install pymupdf
# pip install numpy
# pip install scikit-learn
# pip install nltk
# pip install lxml

# ==================================================================================================
# Subproject I: Web Crawling and Inverted Index Construction
# Following execution, three .json files should be created:
#       document_tokens.json: list of terms lists associated with each crawled PDF
#       inverted_index.json: dictionary mapping each term to its postings list 
#       pdf_metadata.json: list of metadata dictionaries for each PDF (title, degree, year, URLs) 
# ==================================================================================================
import spectrum_spider


# ==================================================================================================
# Subproject II: K-Means Clustering
# Running spectrum_clustering builds TF-IDF matrcies and writes clustering_results.txt.
# This file is split into two parts: global TF-IDF and local TF-IDF. Each part is run with K=[2,10,20].
# ==================================================================================================
import spectrum_clustering


# ============================================================================================================
# Subproject III: Walkthrough of Good and Bad clusters
# These clusters were generated from a previous run with 125 crawled PDFs. 
# The complete output can be found stored in "clustering_results_demo.txt" in the main project directory.
# There is no code is execute from this point onwards, this section exists only to explain clustering results.
# ============================================================================================================

# ====================================================
# GOOD CLUSTER EXAMPLES

# Example #1: Global K = 10, Size = 4, Cluster #0 - Cellular Neuroscience
# Top 10 terms: dopamin, protein, receptor, striatal, amphetamin, locomotor, membran, rat, extracellular
# As observed, terms are coherent and clearly belong to the same academic theme.

# Example #2: Global K = 10, Size = 13, Cluster #3 - Media Studies
# Top 10 terms: artist, aesthet, polit, cultur, discours, contemporari, film, narr, cinema, text
# As observed, terms are coherent and clearly belong to the same academic theme.

# Example #3: Global K = 10, Size = 3, Cluster #7 - Corporate Finance
# Top 10 terms: firm, ceo, equiti, debt, investor, cash, sharehold, stock, fama, ownership
# As observed, terms are coherent and clearly belong to the same academic theme.
# ====================================================

# ====================================================
# BAD CLUSTER EXAMPLES 

# Example #1: Global K = 10, Size = 10, Cluster #2 - ???
# Top 10 terms: erent, rst, cient, ect, ned, signi, max, xed, tion, pro
# None of these are meaningful, and so it is impossible to identify an academic theme.
# They likely represent fractured stems, for instance "erent" = "different" and "rst"= "first".

# Example #2: Global K = 20, Size = 2, Cluster #13 - ???
# Top 10 terms: collus, collud, rm, 2d2, rst, bidder, dea, deb, cartel, plug 
# There's no clear academic theme here, some economic terms but also some malformed stems.
# Because it contains just two documents, their shared noise becomes disproportionately influential.
# An indicator of over-fragmentation, since KMeans is forced to find relationships where they don't exist.

# Example #3: Global K = 20, Size = 1, Cluster #3 - High Performance Computing (HPC)
# Top 10 terms: checkpoint, df, hpc, lc, knapsack, overhead, tsp, tpf, mpi, restart
# Although terms are coherent and belong to the same academic theme, the vocabulary is very jargony.
# An LLM (ChatGPT) was required to interpret the academic theme as I was not personally familiar with the subject matter.
# Clusters such as this are certainly cannot be considered ideal if a human reader cannot understand them.
# Additionally, the size of the cluster is just 1 which clearly indicates over-fragmentation. 
# ====================================================