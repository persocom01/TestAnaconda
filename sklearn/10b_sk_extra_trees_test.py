# ExtraTrees is another take on improving the basic decision tree model. This
# time, not only does it randomize the features in each tree, it also
# randomizes how each decision node in each tree is split. By default though,
# ExtraTrees does not bootstrap the dataset. It is said that ExtraTrees results
# in many more leaves than RandomForest, and works better on noisy datasets,
# but it is uncertain which would perform better without trying them out first.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
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
                          r'[^a-zA-Z ]', str.lower, lup.lemmatize_sentence, ['wa', 'ha'])
print('after:', X[1])
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

sebas = ple.Sebastian()

# ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None,
# min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
# max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
# min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None,
# random_state=None, verbose=0, warm_start=False, class_weight=None,
# ccp_alpha=0.0, max_samples=None)
# The arguments for ExtraTrees are similar to RandomForest, but bootstrap=False
# by default and a random_state argument exists for reproducability.
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('et', ExtraTreesClassifier())
])
params = {
    'tvec__stop_words': ['english'],
    'tvec__ngram_range': [(1, 1), (1, 2)],
    'tvec__max_df': [.3, .6, .9],
    'tvec__min_df': [1, 3, 7],
    'tvec__max_features': [2000, 3000, 4000],
    'et__n_estimators': [100]
}
gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.8419593345656192
print('best score:', gs.best_score_)
# best params: tvec: max_df=0.9, max_features=4000, min_df=2, ngram_range=(1, 1), stop_words='english'
print('best params:', sebas.get_params(gs.best_params_))
print()

tvec = TfidfVectorizer(max_df=0.9, max_features=4000, min_df=2,
                       ngram_range=(1, 1), stop_words='english')
X_train = tvec.fit_transform(X_train)
X_train = pd.DataFrame(X_train.toarray(), columns=tvec.get_feature_names())
X_test = tvec.transform(X_test)
X_test = pd.DataFrame(X_test.toarray(), columns=tvec.get_feature_names())
print('TfidfVectorizer:')
print(X_train.sum().sort_values(ascending=False)[:5])
print()

et = ExtraTreesClassifier(n_estimators=100)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
y_prob = et.predict_proba(X_test)

print('most_important_features:', sebas.get_features(
    X_train, et.feature_importances_))
print()

print('classification report:')
print(classification_report(y_test, y_pred, output_dict=False))
print()

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

yuri = ple.Yuri()

# auc = 0.94
yuri.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))
