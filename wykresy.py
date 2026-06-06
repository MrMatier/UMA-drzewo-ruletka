import os
import pandas as pd
import matplotlib.pyplot as plt

def make_plots(results_path="results/results.csv"):
    if not os.path.exists(results_path):
        print("brak pliku results/results.csv")
        return

    os.makedirs("results/wykresy", exist_ok=True)
    df = pd.read_csv(results_path)

    summary = df.groupby(["dataset", "model"], as_index=False).agg({"accuracy": "mean","precision_macro": "mean","recall_macro": "mean","f1_macro": "mean","depth": "mean","leaves": "mean","train_time": "mean"})
    summary.to_csv("results/summary.csv", index=False)

    for dataset in summary["dataset"].unique():
        data = summary[summary["dataset"] == dataset].copy()
        data = sort_models(data)

        save_bar(data, "accuracy", f"Accuracy - {dataset}", "Accuracy", f"accuracy_{safe_name(dataset)}.png")
        save_bar(data, "f1_macro", f"F1 score - {dataset}", "F1 score", f"f1_{safe_name(dataset)}.png")
        save_bar(data, "depth", f"Tree depth - {dataset}", "Depth", f"depth_{safe_name(dataset)}.png")
        save_bar(data, "leaves", f"Number of leaves - {dataset}", "Leaves", f"leaves_{safe_name(dataset)}.png")

    save_roulette_plots(df)
    print("zapisano wykresy do results/wykresy/")
    print("zapisano podsumowanie do results/summary.csv")

def sort_models(df):
    order = {
        "Greedy tree": 1,
        "Roulette tree": 2,
        "Sklearn tree": 3
    }

    df["order"] = df["model"].map(order)
    df = df.sort_values("order")
    return df.drop(columns=["order"])


def save_bar(df, column, title, ylabel, file_name):
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[column])

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()

    path = os.path.join("results", "wykresy", file_name)
    plt.savefig(path)
    plt.close()


def save_roulette_plots(df):
    roulette = df[df["model"] == "Roulette tree"]

    for dataset in roulette["dataset"].unique():
        data = roulette[roulette["dataset"] == dataset]

        plt.figure(figsize=(8, 5))
        plt.plot(data["seed"], data["accuracy"], marker="o")

        plt.title(f"Roulette accuracy by seed - {dataset}")
        plt.xlabel("Seed")
        plt.ylabel("Accuracy")
        plt.xticks(data["seed"])
        plt.tight_layout()

        path = os.path.join("results", "wykresy", f"roulette_seeds_{safe_name(dataset)}.png")
        plt.savefig(path)
        plt.close()

def safe_name(text):
    return str(text).lower().replace(" ", "_")

if __name__ == "__main__":
    make_plots()