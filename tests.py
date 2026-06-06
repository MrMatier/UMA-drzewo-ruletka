import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from tree_model import DecisionTreeModel
from data import load_datasets

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }

def check_tree(X_train, X_test, y_train, y_test, mode, seed):
    tree = DecisionTreeModel(mode=mode, max_depth=10, min_samples_split=2, random_state=seed)
    start = time.time()
    tree.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = tree.predict(X_test)

    scores = calculate_metrics(y_test, y_pred)

    if mode == "greedy":
        scores["model"] = "Greedy tree"
    else:
        scores["model"] = "Roulette tree"

    scores["seed"] = seed
    scores["depth"] = tree.get_depth()
    scores["leaves"] = tree.count_leaves()
    scores["train_time"] = train_time

    return scores

def check_sklearn(X_train, X_test, y_train, y_test):
    x_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    y_encoder = LabelEncoder()

    X_train_enc = x_encoder.fit_transform(X_train)
    X_test_enc = x_encoder.transform(X_test)

    y_train_enc = y_encoder.fit_transform(y_train)
    y_test_enc = y_encoder.transform(y_test)

    tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
    start = time.time()
    tree.fit(X_train_enc, y_train_enc)
    train_time = time.time() - start

    y_pred_enc = tree.predict(X_test_enc)
    y_pred = y_encoder.inverse_transform(y_pred_enc)

    scores = calculate_metrics(y_test, y_pred)
    scores["model"] = "Sklearn tree"
    scores["seed"] = 42
    scores["depth"] = tree.get_depth()
    scores["leaves"] = tree.get_n_leaves()
    scores["train_time"] = train_time

    return scores

def run_dataset(X, y, dataset_name, roulette_runs=10):
    results = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    greedy_result = check_tree(X_train, X_test, y_train, y_test, mode="greedy", seed=42)
    greedy_result["dataset"] = dataset_name
    results.append(greedy_result)

    for seed in range(roulette_runs):
        roulette_result = check_tree(X_train, X_test, y_train, y_test, mode="roulette", seed=seed)
        roulette_result["dataset"] = dataset_name
        results.append(roulette_result)

    sklearn_result = check_sklearn(
        X_train,
        X_test,
        y_train,
        y_test
    )
    sklearn_result["dataset"] = dataset_name
    results.append(sklearn_result)

    return results

def run_all_tests():
    os.makedirs("results", exist_ok=True)

    datasets = load_datasets()
    all_results = []

    for X, y, name in datasets:
        print(f"Dataset: {name}")

        results = run_dataset(X, y, dataset_name=name, roulette_runs=10)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    cols = ["dataset","model","seed","accuracy","precision_macro","recall_macro","f1_macro","depth","leaves","train_time"]
    df = df[cols]
    df.to_csv("results/results.csv", index=False)

    return df