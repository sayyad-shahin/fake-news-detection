import pytest

from preprocess import clean_text
from predict import MODEL_PATH, predict

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run train.py first")


def test_clean_text():
    assert clean_text("WASHINGTON (Reuters) - Hello, World! http://x.com 123") == "hello world"


@needs_model
def test_predict_shape():
    out = predict("The central bank said inflation rose 3 percent, according to official data.")
    assert out["label"] in {"Real", "Fake"}
    assert abs(out["prob_real"] + out["prob_fake"] - 100) < 0.1


@needs_model
def test_fake_vs_real():
    assert predict("SHOCKING!!! Doctors HATE this miracle cure, share before it gets deleted!!!")["label"] == "Fake"
    assert predict("Officials announced on Monday a new report on housing prices, confirmed by the national statistics office.")["label"] == "Real"


def test_too_short():
    with pytest.raises(ValueError):
        predict("hi")
