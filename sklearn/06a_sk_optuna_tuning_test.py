# Demonstrates nhow optuna is typically used, instead using it like sk-learn
# searches.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import optuna

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline


def objective(trial):
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df.columns = [x.lower().replace(' ', '_') for x in df.columns]
    target = 'target'
    features = [col for col in df.columns if col != target]
    df[target] = data.target
    print(df.head())
    print()

    X = df[features]
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Using any of the special linear regressions requires the features to be
    # scaled. You should also remove features that are highly correlated with each
    # other which was not done here.
    ss = StandardScaler()
    X_train = pd.DataFrame(ss.fit_transform(X_train[features]), columns=features)
    X_test = pd.DataFrame(ss.transform(X_test[features]), columns=features)

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=5)
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
