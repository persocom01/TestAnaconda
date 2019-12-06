import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import_path = r'.\datasets\reddit.csv'
data = pd.read_csv(import_path)
# print(data.columns)
df = data[['title', 'subreddit']]

X = df['title']
y = df['subreddit'].values

contractions = {
    "n't": " not",
    "'s": " is",
    "'m": " am",
    "'ll": " will",
    "'ve": " have",
    "'re'": " are"
    }

# Remove contractions.
for i in range(len(X)):
    for k, v in contractions.items():
        X[i] = re.sub(k, v, X[i])



# Remove slangs.

# Remove punctuation.
for i in range(len(X)):
    X[i] = re.sub(r'[^a-zA-Z ]', r'', X[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# CountVectorizer(input='content', encoding='utf-8', decode_error='strict',
# strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
# stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1),
# analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
# binary=False, dtype=<class 'numpy.int64'>) counts the words in each element
# in a list, and returns a matrix with the rows containing the frequency of
# words in each element
cvec = CountVectorizer(stop_words='english')
X_train_count = cvec.fit_transform(X_train)
X_train_count = pd.DataFrame(X_train_count.toarray(), columns=cvec.get_feature_names())
print(X_train_count.head())

# TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict',
# strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
# analyzer='word', stop_words=None, token_pattern='(?u)\b\w\w+\b',
# ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None,
# binary=False, dtype=<class 'numpy.float64'>, norm='l2', use_idf=True,
# smooth_idf=True, sublinear_tf=False)
tvec = TfidfVectorizer()
# X_tfid =
