# Demonstrates simple text preprocessing in gensim.
import gensim
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pleiades as ple

corpus = [
    "àççéntéd words in english are largely confined to words borrowed from other languages, like résumé and tête-à-tête.",
    "Sugar's is bad to consume. My sister likes to have sugar, but not my father.",
    "My father spends a lot of time driving my sister around to dance practice.",
    "Doctors suggest that driving may cause increased stress and blood pressure.",
    "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better.",
    "Health experts say that Sugar is not good for your lifestyle."
]

# gensim.utils.simple_preprocess(doc, deacc=False, min_len=2, max_len=15)
# In gensim, string documents need to be split into lists of component words,
# and this function does just that while converting all words to lowercase.
# deacc=True converts àççéntéd words to their deaccented counterparts.
# min_len and max_len determines the min and max length of acceptable words.
corpus_split = [gensim.utils.simple_preprocess(doc, deacc=True, min_len=2, max_len=15) for doc in corpus]
print(corpus_split)
