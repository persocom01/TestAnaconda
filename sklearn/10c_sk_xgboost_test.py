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
import xgboost as xgb
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

# xgb(eta=0.3, gamma=0, max_depth=6, min_child_weight=1, max_delta_step=0,
# subsample=1, colsample_bytree=1, colsample_bylevel=1,colsample_bynode=1,
# lambda=1, alpha=0, tree_method='auto', sketch_eps=0.03, scale_pos_weight=1,
# updater=('grow_colmaker', 'prune'), refresh_leaf=1, process_type='default',
# grow_policy='depthwise', max_leaves=0, max_bin=256,predictor='auto',
# num_parallel_tree=1)
# eta=learning_rate

# xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100,
# verbosity=1, objective='binary:logistic', booster='gbtree',
# tree_method='auto', n_jobs=1, gpu_id=-1, gamma=0, min_child_weight=1,
# max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
# colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
# base_score=0.5, random_state=0, missing=None, **kwargs)
# learning_rate=0-1 is the step size.
# max_depth=int is the depth of each decision tree.
# subsample=0-1 works a little like CV in that each iteration would be trained
# on a fraction of the data instead of the whole. Reducing it can prevent
# overfitting.
# colsample_bytree=0-1 is the RandomForest aspect of xgboost in that not all
# features would be used.
# objective sets the loss function. It is automatically set if you call the
# classifier or regressor but other options can be found here:
# https://xgboost.readthedocs.io/en/latest/parameter.html
# gamma, alpha and lambda are regularization parameters. L1 is lasso, lambda is
# ridge.
pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('xgb_class', xgb.XGBClassifier())
])
params = {
    'tvec__stop_words': [None, 'english'],
    'tvec__ngram_range': [(1, 1)],
    'tvec__max_df': [.45, .45, .65],
    'tvec__min_df': [5, 6, 7],
    'tvec__max_features': [200, 300, 400],
}
gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.890018484288355
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


# best params: tvec: max_df=0.65, max_features=300, min_df=5, ngram_range=(1, 1), stop_words=None
print('best params:', get_params(gs.best_params_))
print()

tvec = TfidfVectorizer(max_df=0.65, max_features=300, min_df=5, ngram_range=(1, 1), stop_words=None)
X_train = tvec.fit_transform(X_train)
X_train = pd.DataFrame(X_train.toarray(), columns=tvec.get_feature_names())
X_test = tvec.transform(X_test)
X_test = pd.DataFrame(X_test.toarray(), columns=tvec.get_feature_names())
print('TfidfVectorizer:')
print(X_train.sum().sort_values(ascending=False)[:5])
print()

# Can be done in pipeline but done here as TfidfVectorizer is problem specific.
xgb_class = xgb.XGBClassifier()
params = {
    'max_depth': [1, 3, 5],
    'subsample': [.8, 1],
    'colsample_bytree': [.8, 1]
}
gs = GridSearchCV(xgb_class, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.890018484288355
print('best score:', gs.best_score_)
# best params: {'colsample_bytree': 0.8, 'max_depth': 5, 'subsample': 0.8}
print('best params:', gs.best_params_)
print()

y_pred = gs.predict(X_test)
y_prob = gs.predict_proba(X_test)

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

roc = dp.Roc()
roc.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))

# XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1,
# objective='reg:squarederror', booster='gbtree', tree_method='auto', n_jobs=1,
# gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
# colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0,
# reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0,
# missing=None, num_parallel_tree=1, importance_type='gain', **kwargs
