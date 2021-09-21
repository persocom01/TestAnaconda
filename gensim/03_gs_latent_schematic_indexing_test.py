# Latent schematic indexing (LSI) is an nlp technique used to find similar
# pieces of text. In this example, given a corpus of movie reviews, one can
# find a review most similar to an input string. LSI works by using singular
# value decomposition (SVD) which is non centered PCA. It is good at dealing
# with synonymy and polysemy in languages but is computationally expensive, and
# as such is not recommended for processing documents in bulk.
import pandas as pd
import gensim
from gensim.parsing.preprocessing import preprocess_documents

df = pd.read_csv(‘wiki_movie_plots_deduped.csv’, sep=’,’)
df = df[df[‘Release Year’] >= 2000]
text_corpus = df[‘Plot’].values

processed_corpus = preprocess_documents(text_corpus)
dictionary = gensim.corpora.Dictionary(processed_corpus)
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

tfidf = gensim.models.TfidfModel(bow_corpus, smartirs=’npu’)
corpus_tfidf = tfidf[bow_corpus]

# num_topics is a hyperparameter that can be fine tuned using Topic Coherence measure
lsi = gensim.models.LsiModel(corpus_tfidf, num_topics=200)
index = gensim.similarities.MatrixSimilarity(lsi[corpus_tfidf])

new_doc = gensim.parsing.preprocessing.preprocess_string(new_doc)
new_vec = dictionary.doc2bow(new_doc)
vec_bow_tfidf = tfidf[new_vec]
vec_lsi = lsi[vec_bow_tfidf]
sims = index[vec_lsi]
for s in sorted(enumerate(sims), key=lambda item: -item[1])[:10]:
    print(f”{df[‘Title’].iloc[s[0]]}: {str(s[1])}”)
