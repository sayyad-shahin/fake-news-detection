"""Load the trained model and classify text. Usage: python predict.py "some news text" """
import math
import sys
from pathlib import Path

import joblib
import numpy as np

import preprocess  # noqa: F401  (needed so joblib can unpickle clean_text)

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run `python train.py` first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def _prob_real(model, text: str) -> float:
    clf = model.named_steps["clf"]
    X = model.named_steps["tfidf"].transform([text])
    if hasattr(clf, "predict_proba"):
        return float(clf.predict_proba(X)[0][1])
    return 1 / (1 + math.exp(-float(clf.decision_function(X)[0]) * 2))  # squash margin to 0-1


def _top_words(model, text: str, k: int = 6):
    """Words pushing the prediction toward Real (+) or Fake (-)."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    X = vec.transform([text])
    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        return {"real": [], "fake": []}
    names = vec.get_feature_names_out()
    idx = X.nonzero()[1]
    contrib = sorted(((names[i], float(X[0, i] * weights[i])) for i in idx), key=lambda t: t[1])
    return {"fake": [w for w, c in contrib[:k] if c < 0],
            "real": [w for w, c in reversed(contrib[-k:]) if c > 0]}


def predict(text: str) -> dict:
    text = (text or "").strip()
    if len(text.split()) < 3:
        raise ValueError("Please enter at least a few words of news text.")
    model = get_model()
    p_real = _prob_real(model, text)
    label = "Real" if p_real >= 0.5 else "Fake"
    return {"label": label,
            "confidence": round(max(p_real, 1 - p_real) * 100, 2),
            "prob_real": round(p_real * 100, 2),
            "prob_fake": round((1 - p_real) * 100, 2),
            "key_words": _top_words(model, text)}


if __name__ == "__main__":
    print(predict(" ".join(sys.argv[1:])))
