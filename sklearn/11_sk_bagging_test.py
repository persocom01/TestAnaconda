# Bagging is an ensemble method that first applies bootstrapping (sampling with
# replacement) on the original sample to create new, slighlty different
# samples, running an estimator (default being decision tree) on each of them,
# and averaging the predictions to give a final prediction. This is done to
# simulate running the model on different datasets in order to make it more
# robust.
import pandas as pd
import pleiades as ple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
import data_plots as dp

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
X = cz.text_list_cleaner(X, cz.contractions, reddit_lingo,
                         r'[^a-zA-Z ]', cz.lemmatize_sentence)
print('after:', X[1])
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0,
# max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False,
# warm_start=False, n_jobs=None, random_state=None, verbose=0)
# The main use for BaggingClassifier is running the same model on bootstapped
# datasets.
# base_estimator is by default a decision tree. Different models can be passed,
# for instance, LogisticRegression in this case.
# n_estimators=int determines the number of trees in the forest (or model
# instances)
# max_samples=float_int determines the size of the bootstrapped samples. By
# default it is the same size as the dataset given, but can be made smaller or
# bigger in relation to the dataset if a float is given, or a fixed number if
# an int is given.
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100)
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('bc', BaggingClassifier(base_estimator=lr))
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
# best score: 0.9149722735674677
print('best score:', gs.best_score_)


def get_params(dict):
    from re import match
    params = {}
    pattern = r'^([a-zA-Z0-9_]+)__([a-zA-Z0-9_]+)'
    for k, v in dict.items():
        if isinstance(v, str):
            v = "'" + v + "'"
        m = match(pattern, k)
        key = m.group(1)
        kwarg = f'{m.group(2)}={v}'
        if key in params:
            params[key].append(kwarg)
        else:
            params[key] = [kwarg]
    for k, v in params.items():
        joined_list = ', '.join(map(str, v))
        return f'{k}: {joined_list}'


# best params: tvec: max_df=0.9, max_features=3000, min_df=2, ngram_range=(1, 1), stop_words=None
print('best params:', get_params(gs.best_params_))
print()

tvec = TfidfVectorizer(max_df=0.9, max_features=3000, min_df=2, ngram_range=(1, 1), stop_words=None)
X_train = tvec.fit_transform(X_train)
X_train = pd.DataFrame(X_train.toarray(), columns=tvec.get_feature_names())
X_test = tvec.transform(X_test)
X_test = pd.DataFrame(X_test.toarray(), columns=tvec.get_feature_names())
print('TfidfVectorizer:')
print(X_train.sum().sort_values(ascending=False)[:5])
print()

bc = BaggingClassifier(base_estimator=lr)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
y_prob = bc.predict_proba(X_test)

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

roc = dp.Roc()
roc.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))