# Machine learning models based on decision trees are one of the most effective
# in the industry. Decision trees can be used for both classification and
# regression problems (in practice mostly classification), need no scaling,
# not affected by the skew of the data, are considered fast, and above all,
# are one of the most easily explanable machine learning models.
# However, they will natually overfit the data and need to be pruned, and
# techniques like bagging, random forest (widely used) and extra trees are
# often used to decrease their variance.
import numpy as np
import pandas as pd
import pleiades as ple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
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
X = cz.text_cleaner(X, cz.contractions, reddit_lingo,
                    r'[^a-zA-Z ]', cz.lemmatize_sentence)
print('after:', X[1])
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Before instantiating the model, one must note that classification and
# regression trees use different metrics to determine their splits.
# Classification trees rely on Gini, which is the probability of a random guess
# being wrong, and regression trees use mse.
arr = ['apple', 'banana', 'orange']


def gini(arr):
    probs = []
    for e in set(arr):
        prob = arr.count(e) / len(arr)
        probs.append(prob**2)
    return 1 - sum(probs)


print('Gini:')
print(gini(arr))

# DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
# min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
# max_features=None, random_state=None, max_leaf_nodes=None,
# min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
# ccp_alpha=0.0)
# The most important parameters are max_depth, min_samples_split, and
# min_samples_leaf. Floats between 0 and 1 can be given for the latter 2 to
# make the minimums a proportion rather than a number.
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('dt', DecisionTreeClassifier())
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
# best score: 0.865988909426987
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


# best params: {'tvec__max_df': 0.9, 'tvec__max_features': 2000, 'tvec__min_df': 4, 'tvec__ngram_range': (1, 1), 'tvec__stop_words': None}
# We can use .predict on GridSearchCV instead of reconstructing the model at
# this point, but we'll recreate the model for demonstration purposes.
print('best params:', get_params(gs.best_params_))
print()

tvec = TfidfVectorizer(max_df=0.85, max_features=1000, min_df=2, ngram_range=(1, 2), stop_words=None)
X_train_tvec = tvec.fit_transform(X_train)
X_train_tvec = pd.DataFrame(X_train_tvec.toarray(),
                            columns=tvec.get_feature_names())
X_test_tvec = tvec.transform(X_test)
X_test_tvec = pd.DataFrame(X_test_tvec.toarray(),
                           columns=tvec.get_feature_names())
print('TfidfVectorizer:')
print(X_train_tvec.sum().sort_values(ascending=False)[:5])
print()

# Demonstrates hyperparameter tuning of a decision tree.
max_depths = np.linspace(1, 20, 20)
train_preds = []
test_preds = []
roc = dp.Roc()
for max_depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_train_pred = dt.predict(X_train)
    train_preds.append(roc.auc_score(y_train, y_train_pred))

    y_pred = dt.predict(X_test)
    test_preds.append(roc.auc_score(y_train, y_pred))

preds = [train_preds, test_preds]
roc.plot_auc(max_depths, preds)
