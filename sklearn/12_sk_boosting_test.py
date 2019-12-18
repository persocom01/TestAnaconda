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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
                         r'[^a-zA-Z ]', cz.lemmatize_sentence, ['wa', 'ha'])
print('after:', X[1])
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

sebas = ple.Sebastian()

# AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0,
# algorithm='SAMME.R', random_state=None)
# AdaBoost is short for adaptive boosting. It starts from a base estimator,
# and iterates it n times over the dataset. At each iteration, the weights for
# the incorrectly classified data are increased before the estimator is run
# again, thus allowing the estimator to focus on the incorrectly classified
# cases. The final model comprises a weighted sum of all the estimators.
# AdaBoost uses a decision tree of depth 1 as its base estimator by default.
# If using other estimators, remember to scale when necessary.
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
    'tvec__stop_words': ['english'],
    'tvec__ngram_range': [(1, 1), (1, 2)],
    'tvec__max_df': [.5, .7, .9],
    'tvec__min_df': [2, 4, 6],
    'tvec__max_features': [2000, 3000, 4000],
}
gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
# best score: 0.7569316081330869
print('best score:', gs.best_score_)
# best params: tvec: max_df=0.9, max_features=2000, min_df=2, ngram_range=(1, 1), stop_words='english'
print('best params:', sebas.get_params(gs.best_params_))
print()

tvec = TfidfVectorizer(max_df=0.9, max_features=2000, min_df=2,
                       ngram_range=(1, 1), stop_words='english')
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

# {'china': 0.026813815042157026, 'man': 0.018697088515042638, 'say': 0.018319367054757642, 'russia': 0.017747708591077632, 'right': 0.014303497603874534, 'state': 0.012878299754861385, 'year': 0.011985191727532254, 'trump': 0.011764876191658238,
print('most_important_features:', sebas.get_features(
    X_train, gs.best_estimator_.feature_importances_))
print()

print('classification report:')
print(classification_report(y_test, y_pred, output_dict=False))
print()

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

yuri = ple.Yuri()

# auc = 0.86
yuri.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))
