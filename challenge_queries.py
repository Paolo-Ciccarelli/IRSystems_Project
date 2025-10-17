from query_processor import lookup_singleQ, lookup_andQ
from naive_indexer import inverted_index
from spimi_index import build_spimi_inspired

reuters_dir = 'C:\\Users\\prowl\\Downloads\\reuters21578'  
spimi_inverted_index, _, _ = build_spimi_inspired(reuters_dir)

print("\nChallenge Query #1 'copper':", lookup_singleQ(inverted_index,"copper"))
print("\nChallenge Query #2 'Chrysler':", lookup_singleQ(inverted_index,"Chrysler"))
print("\nChallenge Query #3 'Bundesbank':", lookup_singleQ(inverted_index,"Bundesbank"))

print("\nChallenge Query Naive 'Bundesbank' and 'Chrysler':", lookup_andQ(inverted_index, "Bundesbank", "Chrysler"))
print("\nChallenge Query Naive 'pineapple':", lookup_singleQ(inverted_index, "pineapple"))

print("\nChallenge Query SPIMI 'Bundesbank' and 'Chrysler':", lookup_andQ(spimi_inverted_index, "Bundesbank", "Chrysler"))
print("\nChallenge Query SPIMI 'pineapple':", lookup_singleQ(spimi_inverted_index, "pineapple"))