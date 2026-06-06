import os
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import KBinsDiscretizer

def discretize_data(X, n_bins=3):
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
    X_disc = disc.fit_transform(X)
    X_disc = X_disc.astype(int)
    return X_disc.astype(str)

def load_iris_dataset():
    data = load_iris()
    X = data.data
    y = data.target

    X = discretize_data(X, n_bins=3)
    y = y.astype(str)
    return X, y, "Iris"

def load_breast_cancer_dataset():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X = discretize_data(X, n_bins=3)
    y = y.astype(str)
    return X, y, "Breast_cancer"

def load_mushroom_dataset():
    local_path = os.path.join("data", "mushroom.csv")

    if os.path.exists(local_path):
        df = pd.read_csv(local_path)

        #zakładamy że klasa w pierwszej kolumnie albo nazywa class
        if "class" in df.columns:
            y = df["class"].astype(str).values
            X = df.drop(columns=["class"]).astype(str).values
        else:
            y = df.iloc[:, 0].astype(str).values
            X = df.iloc[:, 1:].astype(str).values

        return X, y, "Mushroom"

    try:
        from ucimlrepo import fetch_ucirepo

        mushroom = fetch_ucirepo(id=73)
        X = mushroom.data.features.astype(str).values
        y = mushroom.data.targets.iloc[:, 0].astype(str).values
        return X, y, "Mushroom"

    except Exception:
        print("nie udało się wczytać Mushroom Dataset")
        print("możesz wrzucić mushroom.csv do folderu data/")
        return None, None, "Mushroom"

def load_datasets():
    datasets = []

    X, y, name = load_mushroom_dataset()
    if X is not None:
        datasets.append((X, y, name))

    datasets.append(load_iris_dataset())
    datasets.append(load_breast_cancer_dataset())
    return datasets