# Latent Semantic Analysis (LSA), also known as Latent Semantic Indexing (LSI),
# is a technique used for dimensionality reduction and semantic analysis of
# text data. LSA is based on the principles of singular value decomposition
# (SVD) and vector space modeling.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords list if not already done
# nltk.download('stopwords')

# Sample documents
documents = [
    'machine learning is great',
    'natural language processing is fun',
    'python is popular',
    'deep learning is exciting',
    'python is useful for machine learning'
]

# Tokenize the documents and remove stopwords
stop_words = set(stopwords.words('english'))

tokenized_docs = []
for doc in documents:
    word_tokens = word_tokenize(doc)
    filtered_words = [word for word in word_tokens if word.lower() not in stop_words]
    tokenized_docs.append(filtered_words)

# Create bigrams
bigram = gensim.models.phrases.Phrases(tokenized_docs, min_count=1, threshold=1)
bigram_phraser = gensim.models.phrases.Phraser(bigram)
tokenized_docs_bigrams = [bigram_phraser[doc] for doc in tokenized_docs]

# Convert tokenized documents to strings
preprocessed_docs = [' '.join(doc) for doc in tokenized_docs_bigrams]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents to TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)

# Specify the number of topics (components)
n_components = 2

# Fit LSA model
lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
lsa_matrix = lsa_model.fit_transform(tfidf_matrix)

# Print the topics
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lsa_model.components_):
    print(f"Topic {topic_idx}:")
    # print top 5 words for each topic
    print([feature_names[i] for i in topic.argsort()[:-6:-1]])
    print()
