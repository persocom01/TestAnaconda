# Demonstrates popular gensim transformations.
import gensim as gs
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pleiades as ple

dict_path = './topic_modeling/docs/dict.mm'
corpus_bow_path = './topic_modeling/docs/corpus_bow.mm'

# Demonstrates loading a saved dictionary and corpus.
dictionary = gs.corpora.Dictionary.load(dict_path)
corpus_bow = gs.corpora.MmCorpus(corpus_bow_path)

# gensim.models.tfidfmodel.TfidfModel(corpus=None, id2word=None,
# dictionary=None, wlocal=<function identity>, wglobal=<function df2idf>,
# normalize=True, smartirs=None, pivot=None, slope=0.25)
# Demonstrates the Term Frequency-Inverse Document Frequency (tfidf) model,
# which penalizes words the more documents they occur in. This is useful as
# words that occur in fewer documents tend to be more useful in
# differentiation.
# dictionary=corpora.Dictionary fits the model using a corpora.Dictionary
# object instead of a bag of words corpus. You still need to apply the model to
# the corpus so it's not very useful.
tfidf = gs.models.TfidfModel(corpus_bow)
corpus_tfidf = tfidf[corpus_bow]

for doc in corpus_tfidf:
    print(doc)

# LSI model algorithm can transform document from either integer valued vector model (such as Bag-of-Words model) or Tf-Idf weighted space into latent space. The output vector will be of lower dimensionality. Following is the syntax of LSI transformation −
lsi = gs.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)

# LDA model algorithm is another algorithm that transforms document from Bag-of-Words model space into a topic space. The output vector will be of lower dimensionality. Following is the syntax of LSI transformation −
# Model=models.LdaModel(corpus, id2word=dictionary, num_topics=100)
#
# RP, a very efficient approach, aims to reduce the dimensionality of vector space. This approach is basically approximate the Tf-Idf distances between the documents. It does this by throwing in a little randomness.
# Model=models.RpModel(tfidf_corpus, num_topics=500)
#
# HDP is a non-parametric Bayesian method which is a new addition to Gensim. We should have to take care while using it.
# Model=models.HdpModel(corpus, id2word=dictionary

# gensim.utils.simple_preprocess(doc, deacc=False, min_len=2, max_len=15)
# In gensim, string documents need to be split into lists of component words,
# and this function does just that while converting all words to lowercase.
# deacc=True converts àççéntéd words to their deaccented counterparts.
# min_len and max_len determines the min and max length of acceptable words.
# corpus_split = [gs.utils.simple_preprocess(doc, deacc=True, min_len=2, max_len=15) for doc in corpus]
# print(corpus_split)
#
# dictionary = gs.corpora.Dictionary(corpus_split)
# print(dictionary.token2id)
