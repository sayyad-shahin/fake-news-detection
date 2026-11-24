"""Text cleaning shared by training and prediction."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<.*?>")
# Agency datelines such as "WASHINGTON (Reuters) -" leak the label in the
# popular Kaggle dataset (all real articles have it), so we strip them.
_DATELINE = re.compile(r"^[A-Z\s,\.]{0,40}\((reuters|ap|afp)\)\s*[-–—]?\s*", re.I)
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    text = str(text)
    text = _DATELINE.sub("", text.strip())
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
