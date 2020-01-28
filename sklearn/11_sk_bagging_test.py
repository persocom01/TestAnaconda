# Bagging is an ensemble method that first applies bootstrapping (sampling with
# replacement) on the original sample to create new, slighlty different
# samples, running an estimator (default being decision tree) on each of them,
# and averaging the predictions to give a final prediction. This is done to
# simulate running the model on different datasets in order to make it more
# robust. Remember to scale the data if the estimator requires scaled data.
import numpy as np
import pandas as pd
import pleiades as ple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
cz = ple.CZ()

print('before:', X[1])
X = cz.text_list_cleaner(X, cz.contractions, reddit_lingo,
                         r'[^a-zA-Z ]', cz.to_lower, cz.lemmatize_sentence, ['wa', 'ha'])
print('after:', X[1])
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

sebas = ple.Sebastian()

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
    ('ss', StandardScaler(with_mean=False)),
    ('bc', BaggingClassifier(base_estimator=lr))
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
# best score: 0.8456561922365989
print('best score:', gs.best_score_)
# best params: tvec: max_df=0.5, max_features=3000, min_df=2, ngram_range=(1, 2), stop_words='english'
print('best params:', sebas.get_params(gs.best_params_))
print()

y_pred = gs.predict(X_test)
y_prob = gs.predict_proba(X_test)

mean_coeff = np.mean([
    model.coef_ for model in gs.best_estimator_.named_steps['bc'].estimators_
], axis=0)

# Bagging does come with a feature_importances_ attribute. For feature
# importances the easiest way is to find feature_importances_ for one of the
# models in the bag.
print('effect of each feature on odds it will be worldnews:')
print(np.exp(mean_coeff))
print()

print('classification report:')
print(classification_report(y_test, y_pred, output_dict=False))
print()

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

yuri = ple.Yuri()

# auc = 0.95
yuri.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))
