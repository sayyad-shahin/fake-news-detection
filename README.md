# 📰 Fake News Detection Using Machine Learning

A complete ML project that classifies news text as **Real** or **Fake**, with a training pipeline, model comparison, a Flask web app, a JSON API and unit tests.

**Tech stack:** Python · scikit-learn · TF-IDF · Logistic Regression / Linear SVM / Naive Bayes · Flask · pandas · pytest

## Features
- Text cleaning pipeline (URLs, HTML, punctuation, agency-dateline leakage removal)
- TF-IDF with unigrams + bigrams
- Trains and compares 3 models, saves the best one by F1-score
- Confusion matrix + metrics saved in `models/`
- Web UI with confidence bar and **explainability** (words pushing toward Real / Fake)
- REST API: `POST /api/predict`
- Unit tests with pytest

## Project structure
```
fake-news-detection/
├── app.py                  # Flask web app + API
├── train.py                # Train, compare, save best model
├── predict.py              # Inference + word-level explanation
├── preprocess.py           # Text cleaning
├── generate_sample_data.py # Creates small demo dataset
├── templates/index.html    # Web UI
├── tests/test_model.py
├── data/                   # Put datasets here
└── models/                 # Saved model + metrics (generated)
```

## Setup
```bash
git clone https://github.com/sayyad-shahin/fake-news-detection.git
cd fake-news-detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset (important)
The repo includes a **small synthetic demo dataset** (`python generate_sample_data.py`) so everything runs immediately. It is too easy, so you will see ~100% accuracy, which is **not** a real-world result.

For real results, download the Kaggle dataset **"Fake and Real News Dataset"** (ISOT, by Clément Bisaillon) and place `Fake.csv` and `True.csv` in `data/`. `train.py` detects them automatically. You can also use your own `data/news.csv` with columns `text,label` (1 = real, 0 = fake).

## Run
```bash
python generate_sample_data.py   # only if using the demo data
python train.py                  # trains and saves models/model.joblib
python app.py                    # open http://127.0.0.1:5000
pytest                           # run tests
python predict.py "Your news text here"   # CLI
```

### API example
```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
     -d '{"text": "Officials announced a new report on inflation, according to the statistics office."}'
```

## Results
After training on the Kaggle dataset, add your numbers here (from `models/metrics.json`):

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | _fill in_ | _fill in_ |
| Linear SVM | _fill in_ | _fill in_ |
| Naive Bayes | _fill in_ | _fill in_ |

## Limitations
The model learns **writing style patterns**, not factual truth. It can be wrong on well-written misinformation or on news from a different domain/time period than its training data. Use it as a learning project, not a fact-checker.

## Future improvements
BERT/transformer fine-tuning · URL input with article scraping · LIME/SHAP explanations · Docker + cloud deployment

## Author
**Shahin Sayyad** · [LinkedIn](https://linkedin.com/in/shahin-sayyad1) · [GitHub](https://github.com/sayyad-shahin)

## License
MIT
