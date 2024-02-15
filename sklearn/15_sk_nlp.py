# demonstrates various text similarity functions:
# 1. cosine_similarity - prefered for entire documents
# 2. euclidean_distances - suitable for comparing texts with lower dimensional
# spaces, such as after techniques like PCA, t-SNE, and SVD are applied.
# 3.jaccard_score - suitable for camparing sets of elements for the presence
# or absence of terms.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances

texts = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# create vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
vectorizer = CountVectorizer(binary=True)
count_matrix = vectorizer.fit_transform(texts)

# compute similarity
cs_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
eu_distances = euclidean_distances(tfidf_matrix, tfidf_matrix)
# there is a need to use .toarray() to convert the count_matrix to a dense
# array
jc_similarities = 1 - pairwise_distances(count_matrix.toarray(), metric='jaccard')

# consine similarity ranges between -1 and 1, with 1 being perfectly similar.
# 0.8, 0.5, 0.2 can be considered high, medium and low.
print('cosine similarities matrix:')
print(cs_similarities)
print()

# euclidean distance is always a positive value, with 0 being perfectly similar
# and larger values being more and more dissimilar.
print('euclidean distance matrix:')
print(eu_distances)
print()

# jaccard similarity ranges between 0 and 1, with 1 being perfectly similar.
print('jaccard similarity matrix:')
print(jc_similarities)
print()

print(f'cosine similarity between document 0 and document 1: {cs_similarities[0, 1]}')
print(f'euclidean distance between document 0 and document 1: {eu_distances[0, 1]}')
print(f'jaccard similarity between document 0 and document 1: {jc_similarities[0, 1]}')
