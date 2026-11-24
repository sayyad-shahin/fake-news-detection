"""Train and compare models, save the best one to models/model.joblib.

Dataset priority:
  1. data/Fake.csv + data/True.csv   (Kaggle "Fake and Real News")
  2. data/news.csv                   (your own CSV with columns: text,label  [1=real, 0=fake])
  3. data/sample_news.csv            (tiny synthetic demo data)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text

DATA, MODELS = Path("data"), Path("models")


def load_data() -> pd.DataFrame:
    if (DATA / "Fake.csv").exists() and (DATA / "True.csv").exists():
        fake, real = pd.read_csv(DATA / "Fake.csv"), pd.read_csv(DATA / "True.csv")
        fake["label"], real["label"] = 0, 1
        df = pd.concat([fake, real], ignore_index=True)
        source = "Kaggle Fake/True"
    elif (DATA / "news.csv").exists():
        df, source = pd.read_csv(DATA / "news.csv"), "data/news.csv"
    else:
        if not (DATA / "sample_news.csv").exists():
            import generate_sample_data  # noqa: F401
            raise SystemExit("Run: python generate_sample_data.py")
        df, source = pd.read_csv(DATA / "sample_news.csv"), "synthetic sample"
    title = df["title"].fillna("") if "title" in df else ""
    df["content"] = (title + " " + df["text"].fillna("")).str.strip()
    df = df[df["content"].str.len() > 0].drop_duplicates("content")
    print(f"Dataset: {source} | {len(df)} articles | real={int(df.label.sum())} fake={int((df.label == 0).sum())}")
    return df


def build(clf) -> Pipeline:
    vec = TfidfVectorizer(preprocessor=clean_text, stop_words="english",
                          ngram_range=(1, 2), max_df=0.7, min_df=2, sublinear_tf=True, max_features=200_000)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def main():
    MODELS.mkdir(exist_ok=True)
    df = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(df["content"], df["label"], test_size=0.2,
                                              random_state=42, stratify=df["label"])
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=5),
        "LinearSVC": LinearSVC(C=1.0),
        "MultinomialNB": MultinomialNB(alpha=0.1),
    }
    results, best_name, best_model, best_f1 = {}, None, None, -1
    for name, clf in candidates.items():
        model = build(clf).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred)
        results[name] = {"accuracy": round(acc, 4), "f1": round(f1, 4)}
        print(f"{name:20s} accuracy={acc:.4f} f1={f1:.4f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    pred = best_model.predict(X_te)
    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_te, pred, target_names=["Fake", "Real"]))
    cm = confusion_matrix(y_te, pred)

    joblib.dump(best_model, MODELS / "model.joblib")
    metrics = {"best_model": best_name, "results": results, "confusion_matrix": cm.tolist(),
               "train_size": len(X_tr), "test_size": len(X_te)}
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Fake", "Real"]); ax.set_yticks([0, 1], ["Fake", "Real"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{best_name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout(); fig.savefig(MODELS / "confusion_matrix.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("Skipped confusion matrix plot:", e)
    print("Saved model to models/model.joblib")


if __name__ == "__main__":
    main()
