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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from io import StringIO
from sklearn.tree import export_graphviz
import pydotplus

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

# Initialize nlp class.
cz = ple.CZ()

print('before:', X[1])
X = cz.text_list_cleaner(X, cz.contractions, reddit_lingo,
                         r'[^a-zA-Z ]', cz.tolower, cz.lemmatize_sentence, ['wa', 'ha'])
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
print()

sebas = ple.Sebastian()

# DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
# min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
# max_features=None, random_state=None, max_leaf_nodes=None,
# min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
# ccp_alpha=0.0)
# The most important parameters are max_depth, min_samples_split, and
# min_samples_leaf. Floats between 0 and 1 can be given for the latter 2 to
# make the minimums a proportion rather than a number.
# criterion='entropy' makes the loss function information gain instead of gini.
# The result is often very similar, but entropy is slightly more
# computationally intensive. Both can be tried to see if one gives better
# results.
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('dt', DecisionTreeClassifier())
])
params = {
    'tvec__stop_words': ['english'],
    'tvec__ngram_range': [(1, 1), (1, 2)],
    'tvec__max_df': [.5, .7, .9],
    'tvec__min_df': [2, 4, 6],
    'tvec__max_features': [2000, 3000, 4000],
}
gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.7689463955637708
print('best score:', gs.best_score_)
# best params: tvec: max_df=0.5, max_features=3000, min_df=2, ngram_range=(1, 2), stop_words='english'
print('best params:', sebas.get_params(gs.best_params_))
print()

tvec = TfidfVectorizer(max_df=0.5, max_features=3000, min_df=2,
                       ngram_range=(1, 2), stop_words='english')
X_train = tvec.fit_transform(X_train)
X_train = pd.DataFrame(X_train.toarray(), columns=tvec.get_feature_names())
X_test = tvec.transform(X_test)
X_test = pd.DataFrame(X_test.toarray(), columns=tvec.get_feature_names())
print('TfidfVectorizer:')
print(X_train.sum().sort_values(ascending=False)[:5])
print()

yuri = ple.Yuri()

# Demonstrates hyperparameter tuning of a decision tree.
# max_depth.
max_depths = np.linspace(1, 10, 10)
auc_scores = yuri.dt_auc_scores(X_train, X_test, y_train, y_test, {
                                'max_depth': max_depths})
yuri.plot_auc(max_depths, auc_scores, title='AUC score vs Tree depth',
              xlabel='Tree depth', labels=['train AUC', 'test AUC'], figsize=(12.5, 7.5))

# min_samples_split.
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
auc_scores = yuri.dt_auc_scores(X_train, X_test, y_train, y_test, {
    'min_samples_split': min_samples_splits})
yuri.plot_auc(min_samples_splits, auc_scores, title='AUC score vs Min samples split',
              xlabel='Min samples split', labels=['train AUC', 'test AUC'], figsize=(12.5, 7.5))

# min_samples_leaf.
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
auc_scores = yuri.dt_auc_scores(X_train, X_test, y_train, y_test, {
    'min_samples_leaf': min_samples_leafs})
yuri.plot_auc(min_samples_leafs, auc_scores, title='AUC score vs Min samples leaf',
              xlabel='Min samples leaf', labels=['train AUC', 'test AUC'], figsize=(12.5, 7.5))

dt = DecisionTreeClassifier()
params = {
    'max_depth': [2, 3, 4],
    'min_samples_split': [0.1, 0.2, 0.3],
    'min_samples_leaf': [0.05, 0.1, 0.15]
}
gs = GridSearchCV(dt, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.5517560073937153
print('best score:', gs.best_score_)
# best params: model args: max_depth=2, min_samples_leaf=0.05, min_samples_split=0.1
print('best params:', sebas.get_params(gs.best_params_))
print()

dt = DecisionTreeClassifier(
    max_depth=2, min_samples_leaf=0.05, min_samples_split=0.1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)

print('most_important_features:', sebas.get_features(
    X_train, dt.feature_importances_))
print()

print('classification report:')
print(classification_report(y_test, y_pred, output_dict=False))
print()

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

# auc = 0.58
yuri.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))

# Demonstrates plotting of a decision tree.
# The tree can be interpreted by getting the most important features from the
# model.
dot_data = StringIO()
# export_graphviz(decision_tree, out_file=None, max_depth=None,
# feature_names=None, class_names=None, label='all', filled=False,
# leaves_parallel=False, impurity=True, node_ids=False, proportion=False,
# rotate=False, rounded=False, special_characters=False, precision=3)
# Generates a graphviz representation of a decision tree.
# feature_names=list names each of the feature with the names in the list.
# This list should contain all feature names, whether they end up appearing in
# the tree or not.
# filled=True makes the node color change depending on the class used for
# classification or the purity of the node in other cases.
# leaves_parallel=True makes the decision tree look more like a flat decision
# caterpillar.
# rounded=True gives the nodes rouded edges and uses Helvetica instead of
# Times-Roman fonts.
# special_characters=False ignores special characters which may be desirable
# for compatibility reasons.
export_graphviz(dt, out_file=dot_data, feature_names=X_train.columns,
                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
export_path = r'.\saved graphs\decision_tree.png'
graph.write_png(export_path)
