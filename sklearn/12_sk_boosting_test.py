# The concept behind the ensemble methods used thus far is to build n
# independent estimators and average their predictions. What boosting does,
# however, is start with a base estimator and sequentially build other
# estimators on top of it to build a powerful ensemble of estimators.
import pandas as pd
import pleiades as ple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

# AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0,
# algorithm='SAMME.R', random_state=None)
# AdaBoost is short for adaptive boosting. It starts from a base estimator,
# and iterates it n times over the dataset. At each iteration, the weights for
# the incorrectly classified data are increased before the estimator is run
# again, thus allowing the estimator to focus on the incorrectly classified
# cases. The final model comprises a weighted sum of all the estimators. In
# a way, since each new iteration seeks a new feature in the dataset to
# classify it by, AdaBoost does something resembling RandomForest.
# AdaBoost uses a decision tree of depth 1 as its base estimator by default.
# n_estimators is a hyperparameter that causes the model to be overfit if it
# is too high.
# learning_rate=float is rate each estimator is shrunk by. The concept is
# similar to alpha in gradient descent; a smaller learning rate equals a
# smaller step, potentially making the solution more accurate at the cost of
# computation time. A small step may, however, cause the model to overfit.
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('ab', AdaBoostClassifier(n_estimators=50))
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
# best score: 0.88909426987061
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


# best params: tvec: max_df=0.85, max_features=1000, min_df=2, ngram_range=(1, 2), stop_words=None
print('best params:', get_params(gs.best_params_))
print()

tvec = TfidfVectorizer(max_df=0.85, max_features=1000, min_df=2,
                       ngram_range=(1, 2), stop_words=None)
X_train = tvec.fit_transform(X_train)
X_train = pd.DataFrame(X_train.toarray(), columns=tvec.get_feature_names())
X_test = tvec.transform(X_test)
X_test = pd.DataFrame(X_test.toarray(), columns=tvec.get_feature_names())
print('TfidfVectorizer:')
print(X_train.sum().sort_values(ascending=False)[:5])
print()

# Can be done in pipeline but done here as TfidfVectorizer is problem specific.
ab = AdaBoostClassifier()
dt = DecisionTreeClassifier
params = {
    'base_estimator': [dt(max_depth=1), dt(max_depth=2)],
    'n_estimators': [30, 50, 70]
}
gs = GridSearchCV(ab, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.8909426987060998
print('best score:', gs.best_score_)
# best params: {'base_estimator': dt(max_depth=1), 'n_estimators': 30}
print('best params:', gs.best_params_)
print()

y_pred = gs.predict(X_test)
y_prob = gs.predict_proba(X_test)

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

roc = dp.Roc()
roc.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))
