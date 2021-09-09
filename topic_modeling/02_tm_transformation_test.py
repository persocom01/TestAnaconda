# Demonstrates popular gensim transformations.
import gensim as gs
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pleiades as ple

corpus_path = './topic_modeling/docs/corpus_bow.mm'

# Demonstrates loading a saved dtm.
corpus_bow = gs.corpora.MmCorpus(corpus_path)

# gensim.models.tfidfmodel.TfidfModel(corpus=None, id2word=None,
# dictionary=None, wlocal=<function identity>, wglobal=<function df2idf>,
# normalize=True, smartirs=None, pivot=None, slope=0.25)
# Demonstrates the Term Frequency-Inverse Document Frequency (tfidf) model.
tfidf = gs.models.TfidfModel(corpus_bow)
corpus_tfidf = tfidf[corpus_bow]

for doc in corpus_tfidf:
    print(doc)

lsi=gs.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)

Model=models.LdaModel(corpus, id2word=dictionary, num_topics=100)

Model=models.RpModel(tfidf_corpus, num_topics=500)

Model=models.HdpModel(corpus, id2word=dictionary

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
