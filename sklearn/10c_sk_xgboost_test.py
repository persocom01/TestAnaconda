# xgboost stands for extreme gradient boosting. Like AdaBoost, it start with a
# base estimator and sequentially build other estimators on top of it. Unlike
# AdaBoost though, it does not reweigh misclassified data after each iteration,
# but instead predicts the residuals or errors of the prior estimators. It
# does not require feature scaling using the default mode, but if the mode is
# changed to gblinear it may need it. xgboost can be used to feature select for
# itself.
import pandas as pd
import pleiades as ple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import_path = r'.\datasets\reddit.csv'
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

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

sebas = ple.Sebastian()

# xgb(eta=0.3, gamma=0, max_depth=6, min_child_weight=1, max_delta_step=0,
# subsample=1, colsample_bytree=1, colsample_bylevel=1,colsample_bynode=1,
# lambda=1, alpha=0, tree_method='auto', sketch_eps=0.03, scale_pos_weight=1,
# updater=('grow_colmaker', 'prune'), refresh_leaf=1, process_type='default',
# grow_policy='depthwise', max_leaves=0, max_bin=256, predictor='auto',
# num_parallel_tree=1)
# eta=0-1 is the learning_rate or step size.
# max_depth=int is the depth of each decision tree.
# subsample=0-1 works a little like CV in that each iteration would be trained
# on a fraction of the data instead of the whole. Reducing it can prevent
# overfitting.
# colsample_bytree=0-1 is the RandomForest aspect of xgboost in that not all
# features would be used.
# gamma, alpha and lambda are regularization parameters. L1 is lasso, lambda is
# ridge.

# xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100,
# verbosity=1, objective='binary:logistic', booster='gbtree',
# tree_method='auto', n_jobs=1, gpu_id=-1, gamma=0, min_child_weight=1,
# max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
# colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
# base_score=0.5, random_state=0, missing=None, **kwargs)
# objective sets the loss function. It is automatically set for the base xgb
# objectif you call the classifier or regressor but other options can be found
# here: https://xgboost.readthedocs.io/en/latest/parameter.html
# subsample=0-1 if < 1, enables random sampling of the train set.
# colsample_bytree=0-1 if < 1, enables random sampling of feature columns. For
# comparison, in RandomForest, max_features=sqrt(n_features).
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('xgb_class', xgb.XGBClassifier())
])
params = {
    'tvec__stop_words': ['english'],
    'tvec__ngram_range': [(1, 1), (1, 2)],
    'tvec__max_df': [.3, .6, .9],
    'tvec__min_df': [1, 3, 7],
    'tvec__max_features': [1000, 1500, 2000],
    'xgb_class__max_depth': [4, 5, 6],
    'xgb_class__subsample': [.8, 1],
    'xgb_class__colsample_bytree': [.8, 1]
}
gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.7458410351201479
print('best score:', gs.best_score_)
# best params: tvec: max_df=0.1, max_features=1500, min_df=2, ngram_range=(1, 2), stop_words='english'
print('best params:', sebas.get_params(gs.best_params_))
print()

tvec = TfidfVectorizer(max_df=0.1, max_features=1500, min_df=2,
                       ngram_range=(1, 2), stop_words='english')
X_train = tvec.fit_transform(X_train)
# Remove whitespace in feature names to prevent xgb.plot_tree from breaking
# later.
feature_names = [x.replace(' ', '_') for x in tvec.get_feature_names()]
X_train = pd.DataFrame(X_train.toarray(), columns=feature_names)
X_test = tvec.transform(X_test)
X_test = pd.DataFrame(X_test.toarray(), columns=feature_names)
print('TfidfVectorizer:')
print(X_train.sum().sort_values(ascending=False)[:5])
print()

# Can be done in pipeline but done here as TfidfVectorizer is problem specific.
# As the xgb object itself contains several useful methods the model will be
# reinitialized after the ideal parameters are found.
xgb_class = xgb.XGBClassifier()
params = {
    'max_depth': [4, 5, 6],
    'subsample': [.8, 1],
    'colsample_bytree': [.8, 1]
}
gs = GridSearchCV(xgb_class, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.777264325323475
print('best score:', gs.best_score_)
# best params: model args: colsample_bytree=0.8, max_depth=6, subsample=0.8
print('best params:', sebas.get_params(gs.best_params_))
print()

xgb_class = xgb.XGBClassifier(colsample_bytree=0.8, max_depth=6, subsample=0.8)
xgb_class.fit(X_train, y_train)

y_pred = xgb_class.predict(X_test)
y_prob = xgb_class.predict_proba(X_test)

print('most_important_features:', sebas.get_features(
    X_train, xgb_class.feature_importances_))
print()

print('classification report:')
print(classification_report(y_test, y_pred, output_dict=False))
print()

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

yuri = ple.Yuri()

# auc = 0.90
yuri.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))

# xgb.plot_tree breaks if feature names contain whitespaces.
xgb.plot_tree(xgb_class, num_trees=0)
plt.rcParams['figure.figsize'] = [12, 7.5]
plt.rcParams['figure.dpi'] = 1000
plt.savefig('./saved graphs/xgb_tree.jpg', dpi=1000)
plt.show()
plt.close()

# xgb.plot_importance(booster, ax=None, height=0.2, xlim=None, ylim=None,
# title='Feature importance', xlabel='F score', ylabel='Features',
# importance_type='weight', max_num_features=None, grid=True,
# show_values=True, **kwargs)
# importance_type accepts 5 possible arguments:  'gain', 'weight', 'cover',
# 'total_gain' or 'total_cover'. Note that feature_importances_ uses 'gain' by
# default, so to sync the two results this has to be set to 'gain'.
# max_num_features=int needs to be set if you don't want to to end up with an
# unreadable graph.
xgb.plot_importance(xgb_class, importance_type='gain', max_num_features=10)
plt.show()
plt.close()

# XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1,
# objective='reg:squarederror', booster='gbtree', tree_method='auto', n_jobs=1,
# gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
# colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0,
# reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0,
# missing=None, num_parallel_tree=1, importance_type='gain', **kwargs
