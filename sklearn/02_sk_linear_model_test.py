import pandas as pd
import sklearn.datasets as skds
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
import sklearn.metrics as skm

data = skds.load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())
print()

# Typically, if the DataFrame is not already split into features and target,
# you can get a list of feature names with the following code, before passing
# it to the DataFrame as a list of columns:
# features = [col for col in df.columns if col != 'target']
X = df
y = data.target

# skms.train_test_split(arr_features, arr_target, test_size=0.25, **options)
# options accepts a number of arguments, including:
# test_size=float_int if given an int, it takes it as the absolute number of
# rows to take as the test data. If given a float, takes it as the proportion
# of the total data to take as the test set.
# random_state=int is the rng seed.
# shuffle=True randomizes the data before splitting.
# stratify=list lets you control the proportion of values in the split by
# specifying a categorical value for each row in the array. The split will
# have the same proportion of these categorical values. In practice you just
# pass it one of the DataFrame's categorial columns.
# In this case, the 'age' column was effectively made a categorical column
# where the proportion of 'age' below 0 is the same in the training and
# testing split.
X_train, X_test, y_train, y_test = skms.train_test_split(X, y, stratify=(df['age'] < 0))
print('train:')
print(X_train.head())
print('count:', len(X_train))
print()
print('test:')
print(X_test.head())
print()

# Demonstrates training a model on test data.
# sklm.LinearRegression(fit_intercept=True, normalize=False, copy_X=True,
# n_jobs=None)
# sklm.LinearRegression() needs to be called again for every new model.
# normalize=True will deduct the mean from each value and divide it by the
# l2 norm, which is the root of the sum of the squares of all values.
# lr.fit(self, X, y, sample_weight=None)
lm = sklm.LinearRegression().fit(X_train, y_train)

# lm.predict(self, X) returns y values predicted by the model for input values
# of X. We normally pass it the X_test values because we want to see how close
# predicted y is to y_test.
lm_predict = lm.predict(X_test)
print('predictions:')
print(lm_predict[:10])
print('count:', len(lm_predict))
print()

# skm.r2_score(y_true, y_pred, sample_weight=None,
# multioutput=’uniform_average) returns the R**2 value of the prediction,
# where:
# R**2 = 1 - (sum of squared residuals) / (total sum of squares)
# residuals being (prediction - mean)**2,
# and squares being (actual value - mean)**2.
# The maximum value of R**2 is 1.0, but can be negative, if the model performs
# worse than the mean.
print('R^2:')
print(skm.r2_score(y_test, lm_predict))
print()
