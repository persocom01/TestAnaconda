# Naive bayes is a model used for binary classification problems. It is fast
# and requires little training data, making it applicable for making real time
# predictions. However, it is known as a bad estimator, so predict_proba isn't
# too reliable, and it runs on the assumption that its features are completely
# independent.
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# For binary features.
# from sklearn.naive_bayes import BernoulliNB
# For discrete features.
from sklearn.naive_bayes import MultinomialNB
# This is MultinomialNB but instead of taking into account the probability of
# an element being in a class, it calculates the probability of the element
# occuring in other classes, then takes the complement. (1-P) The lowest result
# is the predicted class.
# from sklearn.naive_bayes import ComplementNB
# For features that follow a normal distribution.
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import sys
sys.path.append('..')
import pleiades as ple

import_path = r'./datasets/reddit.csv'
data = pd.read_csv(import_path, index_col=None)
# print(data.columns)
df = data[['title', 'subreddit']]

X = df['title']
y = df['subreddit'].values

# Lingo dict.
# TIL is here because leaving it in makes it too easy.
reddit_lingo = {
    'TIL': '',
    '[tT]oday [iI] [lL]earned': '',
    'ff+uu+': 'ffuuu'
}

# Initialize nlp class.
lup = ple.Lupu()

print('before:', X[1])
X = lup.text_list_cleaner(X, lup.contractions, reddit_lingo,
                          r'[^a-zA-Z ]', lup.to_lower, lup.lemmatize_sentence, ['wa', 'ha'])
print('after:', X[1])
print()

full_text = ' '.join(X)
lup.word_cloud(full_text, background_color='white', colormap='nipy_spectral_r')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, stratify=y)

sebas = ple.Sebastian()

# At this point one can use pipe and GridSearchCV to find the best parameters:
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
params = {
    'tvec__stop_words': ['english'],
    'tvec__ngram_range': [(1, 1), (1, 2)],
    'tvec__max_df': [.3, .6, .9],
    'tvec__min_df': [1, 3, 7],
    'tvec__max_features': [2000, 3000, 4000],
}
gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.8585951940850277
print('best score:', gs.best_score_)
# best params: tvec: max_df=0.5, max_features=3000, min_df=2, ngram_range=(1, 2), stop_words='english'
print('best params:', sebas.get_params(gs.best_params_))
print()

# CountVectorizer(input='content', encoding='utf-8', decode_error='strict',
# strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
# stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1),
# analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
# binary=False, dtype=<class 'numpy.int64'>) counts the words in each element
# in a list, and returns a matrix with the rows being the frequency of words in
# each element.
# stop_words='english' works for most cases but you can pass your own list.
# ngram_range=(min_words, max_words) determines the min and max length of each
# word feature.
# max / min_df=0-1_int ignores terms that occur in more or less than float
# proportion of documents or more or less than int number of times.
# max_features=int self explanatory, but know that it eliminates features
# with the smallest column sums first.
# Stopwords can be removed before during the vectorization stage, but
# I've found it computationally expensive. If you wish to do it remember to add
# \b to each side of the word during regex.
print('stopwords:', stopwords.words('english'))
print()
cvec = CountVectorizer(max_df=0.5, max_features=3000, min_df=2,
                       ngram_range=(1, 2), stop_words='english')
X_train_cvec = cvec.fit_transform(X_train)
X_train_cvec = pd.DataFrame(X_train_cvec.toarray(),
                            columns=cvec.get_feature_names())
X_test_cvec = cvec.transform(X_test)
X_test_cvec = pd.DataFrame(X_test_cvec.toarray(),
                           columns=cvec.get_feature_names())
print('CountVectorizer:')
print(X_train_cvec.sum().sort_values(ascending=False)[:5])
print()

# TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict',
# strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
# analyzer='word', stop_words=None, token_pattern='(?u)\b\w\w+\b',
# ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None,
# binary=False, dtype=<class 'numpy.float64'>, norm='l2', use_idf=True,
# smooth_idf=True, sublinear_tf=False) is a kind of CountVectorizer that
# penalizes words that occur too often and boosts words that occur less often.
# In practice it often produces much better results than CountVectorizer.
# The equivalent of using CountVectorizer() followed by TfidfTransformer().
tvec = TfidfVectorizer(max_df=0.5, max_features=3000, min_df=2,
                       ngram_range=(1, 2), stop_words='english')
X_train_tvec = tvec.fit_transform(X_train)
X_train_tvec = pd.DataFrame(X_train_tvec.toarray(),
                            columns=tvec.get_feature_names())
X_test_tvec = tvec.transform(X_test)
X_test_tvec = pd.DataFrame(X_test_tvec.toarray(),
                           columns=tvec.get_feature_names())
print('TfidfVectorizer:')
print(X_train_tvec.sum().sort_values(ascending=False)[:5])
print()

nb = MultinomialNB()
model = nb.fit(X_train_cvec, y_train)
model_tvec = nb.fit(X_train_tvec, y_train)

y_pred = model.predict(X_test_cvec)
y_pred_tvec = model_tvec.predict(X_test_tvec)
y_prob = model.predict_proba(X_test_cvec)
y_prob_tvec = model_tvec.predict_proba(X_test_tvec)

print('model accuracy on itself:')
print('cvec:', model.score(X_train_cvec, y_train))
print('tvec:', model_tvec.score(X_train_tvec, y_train))
print()
print('model accuracy on test set:')
print('cvec:', model.score(X_test_cvec, y_test))
# tvec: 0.8864265927977839
print('tvec:', model_tvec.score(X_test_tvec, y_test))
print()

yuri = ple.Yuri()

print('classification report:')
print(classification_report(y_test, y_pred, output_dict=False))
print()

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

# auc = 0.96
yuri.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))

# Demonstrates how to save models for use later.
# pickle the vectorizer.
pickle.dump(tvec, open(r'.\pickles\vector_nb_tvec.sav', 'wb'))
# pickle the model.
pickle.dump(nb, open(r'.\pickles\model_nb.sav', 'wb'))

# Code on how to load the saved models.
# with open(r'.\pickles\vector_nb_tvec.sav', 'rb') as f:
#     tvec = pickle.load(f)
