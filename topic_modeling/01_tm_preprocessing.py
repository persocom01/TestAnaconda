# Demonstrates common preprocessing steps before using gensim models.
from nltk.corpus import stopwords
import gensim as gs
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pleiades as ple

corpus = []
dir = './topic_modeling/docs'
corpus_path = './topic_modeling/docs/corpus_bow.mm'


# Demonstrates reading all files in a folder.
def read_files(dir):
    import glob
    for file in glob.glob(f'{dir}/*.txt'):
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.read().splitlines():
                yield line


for line in read_files(dir):
    corpus.append(line)

lup = ple.Lupu()

corpus_clean = lup.text_list_cleaner(corpus, str.lower, lup.contractions, lup.remove_numbers, lup.lemmatize_sentence, stopwords.words('english'), lup.remove_extra_spaces)

# gensim.utils.simple_preprocess(doc, deacc=False, min_len=2, max_len=15)
# In gensim, string documents need to be split into lists of component words,
# and this function does just that while converting all words to lowercase and
# removing punctuation.
# deacc=True converts àççéntéd words to their deaccented counterparts.
# min_len and max_len determines the min and max length of acceptable words.
corpus_split = [gs.utils.simple_preprocess(doc, deacc=True, min_len=2, max_len=15) for doc in corpus_clean]
print(corpus_split)
print()

# gensim.corpora.dictionary.Dictionary(documents=None, prune_at=2000000)
# gensim models may require this dictionary to be passed as the id2word
# argument. What the dictionary does is map each word, known as a token in
# gensim, to an integer id. We normally do not use the dictionary alone, but it
# contains several properties that can be useful:
dictionary = gs.corpora.Dictionary(corpus_split)
print('number of tokens: ' + str(dictionary))
print('token to id mapping:')
print(dictionary.token2id)
print('id to frequency of occurance:')
print(dictionary.cfs)
print('number of documents containing each token')
print(dictionary.dfs)
print('number of documents processed: ' + str(dictionary.num_docs))
print('number of processed words: ' + str(dictionary.num_pos))
print()

# doc2bow(document, allow_update=False, return_missing=False)
# Once the dictionary is made, we convert it to a bag of words doc term matrix
# using doc2bow.
corpus_bow = [dictionary.doc2bow(doc, allow_update=True) for doc in corpus_split]
print(corpus_bow)
# The above is the typical input for gensim transformations. But for human
# readability, use the following code:
corpus_bow_readable = [[(dictionary[id], count) for id, count in line] for line in corpus_bow]
print(corpus_bow_readable)

# Demonstrates saving the doc term matrix.
gs.corpora.MmCorpus.serialize(corpus_path, corpus_bow)
