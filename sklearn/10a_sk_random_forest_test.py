# RandomForest is a tried and tested algorithm and can be said to be an
# improvement over the basic decision tree. It attempts to correct for the
# decision tree's tendency to always favor the best predictor, resulting in
# similar trees if a single feature overshadows the others. RandomForest
# comprises trees all built with a different subset of features and coming up
# with a mean prediction from all the trees. It can be said to be a special
# bagging ensemble of decision trees.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os, sys
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

# RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
# min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
# max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
# random_state=None, verbose=0, warm_start=False, class_weight=None,
# ccp_alpha=0.0, max_samples=None)
# The arguments for RandomForest are similar to base decision trees, but it
# should be noted that due to the smaller set of features per tree, the optimal
# max depth is often much deeper, so much so that it is uncertain if there is a
# point in searching for it.
# max_features determines the number of features per tree. The default is
# sqrt(n_features) but a float between 0 and 1, ints or 'log2' can also be
# used.
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('rf', RandomForestClassifier())
])
params = {
    'tvec__stop_words': ['english'],
    'tvec__ngram_range': [(1, 1), (1, 2)],
    'tvec__max_df': [.3, .6, .9],
    'tvec__min_df': [1, 3, 7],
    'tvec__max_features': [2000, 3000, 4000],
    'rf__n_estimators': [100],
    'rf__max_depth': [None],
}
gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.833641404805915
print('best score:', gs.best_score_)
# best params: tvec: max_df=0.7, max_features=3000, min_df=2, ngram_range=(1, 2), stop_words='english'
print('best params:', sebas.get_params(gs.best_params_))
print()

tvec = TfidfVectorizer(max_df=0.7, max_features=3000, min_df=2,
                       ngram_range=(1, 2), stop_words='english')
X_train = tvec.fit_transform(X_train)
X_train = pd.DataFrame(X_train.toarray(), columns=tvec.get_feature_names())
X_test = tvec.transform(X_test)
X_test = pd.DataFrame(X_test.toarray(), columns=tvec.get_feature_names())
print('TfidfVectorizer:')
print(X_train.sum().sort_values(ascending=False)[:5])
print()

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

print('most_important_features:', sebas.get_features(
    X_train, rf.feature_importances_))
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
