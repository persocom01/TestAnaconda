# Latent Dirichlet Allocation (LDA) is the most common topic modeling nlp
# technique. Topic modelling is an unsupervised learning approach in which the
# model identifies the topics by detecting the patterns such as words clusters.
# The outputs of a topic model are:
# 1. clusters of documents that the model has grouped based on topics
# 2. clusters of words (topics) that the model has used to infer the relations.
import gensim
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords list if not already done
# nltk.download('stopwords')
# nltk.download('punkt')

# Sample documents
documents = [
    'machine learning is great',
    'natural language processing is fun',
    'python is popular',
    'deep learning is exciting',
    'python is useful for machine learning'
]

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))

tokenized_docs = []
for doc in documents:
    if doc is None:
        tokenized_docs.append([])
    else:
        # remove punctuation
        doc = doc.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        word_tokens = word_tokenize(doc)
        filtered_words = [word for word in word_tokens if word.lower() not in stop_words]
        tokenized_docs.append(filtered_words)

# Create bigrams
bigram = gensim.models.phrases.Phrases(tokenized_docs, min_count=1, threshold=1)
bigram_phraser = gensim.models.phrases.Phraser(bigram)
tokenized_docs_bigrams = [bigram_phraser[doc] for doc in tokenized_docs]

# Create a dictionary representation of the documents
dictionary = gensim.corpora.Dictionary(tokenized_docs_bigrams)

# Convert the documents to a bag-of-words corpus
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs_bigrams]

# Build the LDA model
n_topics = 2
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, passes=15)

# Print the topics
for i in range(n_topics):
    print(f'topic {i}:')
    # get top 3 topics
    for topic in lda_model.get_topic_terms(i, topn=3):
        print(f'{dictionary.get(topic[0])} {topic[1]}')
