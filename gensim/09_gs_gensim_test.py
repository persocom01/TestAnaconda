from nltk.corpus import stopwords
import gensim as gs
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pleiades as ple

corpus = [
    "Sugar is bad to consume. My sister likes to have sugar, but not my father.",
    "My father spends a lot of time driving my sister around to dance practice.",
    "Doctors suggest that driving may cause increased stress and blood pressure.",
    "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better.",
    "Health experts say that Sugar is not good for your lifestyle."
]

lup = ple.Lupu()

corpus_clean = lup.text_list_cleaner(corpus, lup.contractions, lup.remove_numbers, lup.lemmatize_sentence, ['wa', 'ha'], stopwords.words('english'), lup.remove_extra_spaces)
corpus_split = [gs.utils.simple_preprocess(doc, deacc=True, min_len=2, max_len=15) for doc in corpus_clean]
dictionary = gs.corpora.Dictionary(corpus_split)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus_split]

lda = gs.models.ldamodel.LdaModel
ldamodel = lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=3))
