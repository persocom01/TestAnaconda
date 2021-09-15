# SVM or Support Vector Machines is a complicated algorithm that can be used
# for both regression and classification problems (in practice mostly
# classification). It excels in handling datasets with large numbers of
# features, is highly accurate and is robust to outliers. However, it is noise
# sensitive (if not properly regularized), does not provide probability
# estimates, and is slow in large datasets. It is used in image classification,
# text mining classification and gene expression classification. It can also be
# used for outlier detection and clustering.
import pandas as pd
import pleiades as ple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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
lup = ple.Lupu()

print('before:', X[1])
X = lup.corpus_cleaner(X, str.lower, lup.contractions, reddit_lingo,
                          r'[^a-zA-Z ]', lup.lemmatize_sentence, ['wa', 'ha'])
print('after:', X[1])
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

sebas = ple.Sebastian()

# SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True,
# probability=False, tol=0.001, cache_size=200, class_weight=None,
# verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False,
# random_state=None)
# C=float is the cost of misclassification. A higher value will result in a
# greater likelyhood of overfitting.
# kernel=str can be one of 5 values, or a collable. The values are 'linear',
# 'poly', 'rbf', 'sigmoid', and 'precomputed'. The default value is 'rbf', but
# for nlp classification problems with only two target classes, we can go with
# 'linear', which is meant for classes which are linearly separable (always
# the case when there are only 2 classes).
# gamma=float is the kernel coefficient. It controls how aggressively the
# kernel will try to fit points via higher dimensional transformation. A larger
# value makes the kernel less aggressive and increases the likelyhood of
# underfitting. Manual values of gamma normally range take the range
# {'gamma':np.logspace(-5, -1, 10)}
# pipe = Pipeline([
#     ('tvec', TfidfVectorizer()),
#     ('svc', SVC(gamma='scale'))
# ])
# params = {
#     'tvec__stop_words': ['english'],
#     'tvec__ngram_range': [(1, 1), (1, 2)],
#     'tvec__max_df': [.5, .7, .9],
#     'tvec__min_df': [2, 4, 6],
#     'tvec__max_features': [2000, 3000, 4000],
#     'svc__C': [1, 4, 7],
#     'svc__kernel': ['linear'],
# }
# gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
# gs.fit(X_train, y_train)
# # best score: 0.8604436229205176
# print('best score:', gs.best_score_)
# # best params: svc: C=1, kernel='linear' tvec: max_df=0.5, max_features=2000, min_df=2, ngram_range=(1, 1), stop_words='english'
# print('best params:', sebas.get_params(gs.best_params_))
# print()

tvec = TfidfVectorizer(max_df=0.5, max_features=2000, min_df=2,
                       ngram_range=(1, 1), stop_words='english')
X_train = tvec.fit_transform(X_train)
X_train = pd.DataFrame(X_train.toarray(), columns=tvec.get_feature_names())
X_test = tvec.transform(X_test)
X_test = pd.DataFrame(X_test.toarray(), columns=tvec.get_feature_names())
print('TfidfVectorizer:')
print(X_train.sum().sort_values(ascending=False)[:5])
print()

# We reinitialize the model here because we don't know how to get the
# coefficients from the gridearch object.
svc = SVC(gamma='scale', C=1, kernel='linear')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

# Only usable if SVC kernel='linear'.
print('most_important_features:', sebas.get_features(X_train, svc.coef_.ravel()))
print()

sebas.plot_importances(order='dsc')
sebas.plot_importances(order='asc')
sebas.plot_importances(order='abs')

print('classification report:')
print(classification_report(y_test, y_pred, output_dict=False))
print()

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()
