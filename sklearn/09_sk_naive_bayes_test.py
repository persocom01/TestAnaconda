# Naive bayes is a model used for binary classification problems. It is fast
# and requires little training data, making it applicable for making real time
# predictions. However, it is known as a bad estimator, so predict_proba isn't
# too reliable, and it runs on the assumption that its features are completely
# independent.
import pandas as pd
import pleiades as ple
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import data_plots as dp
# For binary features.
# from sklearn.naive_bayes import BernoulliNB
# For discrete features.
from sklearn.naive_bayes import MultinomialNB
# For features that follow a normal distribution.
# from sklearn.naive_bayes import GaussianNB


import_path = r'.\datasets\reddit.csv'
data = pd.read_csv(import_path)
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

cz = ple.CZ()
print('before:', X[1])
X = cz.text_cleaner(X, cz.contractions, reddit_lingo,
                    r'[^a-zA-Z ]', cz.lemmatize_sentence)
print('after:', X[1])
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, stratify=y)

# At this point one can use pipe and GridSearchCV to find the best parameters:
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
params = {
    'tvec__stop_words': [None, 'english'],
    'tvec__ngram_range': [(1, 1), (1, 2)],
    'tvec__max_df': [.85, .9, .95],
    'tvec__min_df': [2, 4, 6],
    'tvec__max_features': [1000, 2000, 3000],
}
gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.9316081330868762
print('best score:', gs.best_score_)
# best params: {'tvec__max_df': 0.85, 'tvec__max_features': 3000, 'tvec__min_df': 2, 'tvec__ngram_range': (1, 1), 'tvec__stop_words': None}
print('best params:', gs.best_params_)
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
# max / min_df=float_int ignores terms that occur in more or less than float
# proportion of documents or more or less than int number of times.
# max_features=int self explanatory, but know that it eliminates features
# with the smallest column sums first.
# Stopwords can be removed before during the vectorization stage, but
# I've found it computationally expensive. If you wish to do it remember to add
# \b to each side of the word during regex.
print('stopwords:', stopwords.words('english'))
print()
cvec = CountVectorizer(stop_words=None, ngram_range=(
    1, 1), max_df=0.85, min_df=2, max_features=3000)
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
tvec = TfidfVectorizer(stop_words=None, ngram_range=(
    1, 1), max_df=0.85, min_df=2, max_features=3000)
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

print('model accuracy on itself:')
print('cvec:', model.score(X_train_cvec, y_train))
print('tvec:', model_tvec.score(X_train_tvec, y_train))
print()
print('model accuracy on test set:')
print('cvec:', model.score(X_test_cvec, y_test))
print('tvec:', model_tvec.score(X_test_tvec, y_test))
print()

roc = dp.Roc()
roc.plot(y_test, y_pred, figsize=(12.5, 7.5))
roc.plot(y_test, y_pred_tvec, figsize=(12.5, 7.5))
