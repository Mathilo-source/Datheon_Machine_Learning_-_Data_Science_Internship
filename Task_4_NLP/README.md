# NLP Sentiment Classification on Customer Reviews (TF-IDF)

A text classifier that labels reviews as **positive** or **negative**. It cleans raw text, converts it to TF-IDF features, and trains two models (Multinomial Naive Bayes and Logistic Regression) for comparison. It includes a `predict_sentiment()` helper for testing your own phrases.

## Contents

| File | Description |
|---|---|
| `sentiment_classification.ipynb` | The full pipeline, with outputs already executed |
| `sentiment_reviews.csv` | Provided starter file (10 rows), used as a sanity check |
| `README.md` | This file |

## Quick start

```bash
pip install nltk scikit-learn pandas matplotlib jupyter
jupyter notebook sentiment_classification.ipynb
```

Place `sentiment_reviews.csv` in the same folder as the notebook. The first cell downloads the required NLTK data (`stopwords`, `wordnet`, `omw-1.4`, `movie_reviews`) automatically, so an internet connection is needed on first run.

## Data

- **Starter file:** `sentiment_reviews.csv` has columns `review_id`, `review_text`, `sentiment`. It contains only 10 rows (5 positive, 4 negative, 1 neutral), which is too small to train or benchmark a model reliably.
- **Training corpus:** NLTK's `movie_reviews` (2,000 labelled reviews, balanced 1,000 / 1,000).
- **How each is used:** the model is trained and benchmarked on the movie reviews. The 9 positive/negative starter rows serve as a small out-of-domain product-review check. The neutral row is dropped because the task is binary.

## Pipeline

1. **Preprocessing:** strip HTML tags, lowercase, expand contractions, remove punctuation and digits, tokenize, remove stopwords, lemmatize (NLTK `WordNetLemmatizer`). Negation words (`not`, `no`, `nor`, `never`) are deliberately **kept**, since they flip sentiment.
2. **Features:** `TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)`, so bigrams such as "not good" become features.
3. **Models:** `MultinomialNB` and `LogisticRegression`, each in a scikit-learn `Pipeline` with the vectorizer. The vectorizer is fit on training data only, so nothing leaks from the test set.
4. **Evaluation:** an 80/20 stratified split with classification reports and confusion matrices, plus 5-fold cross-validation.
5. **Interactive helper:** `predict_sentiment()`.

## Results

Held-out test set (400 reviews):

| Model | Test accuracy | 5-fold CV (mean ± std) |
|---|---|---|
| Multinomial Naive Bayes | 81.0% | 82.6% ± 1.0% |
| Logistic Regression | 85.8% | 86.7% ± 1.2% |

Out-of-domain check on the 9 starter product reviews: Naive Bayes 78%, Logistic Regression 67%. With only 9 samples this is a rough signal, not a benchmark.

Exact figures can vary slightly across scikit-learn or NLTK versions.

## Using the test helper

```python
predict_sentiment("What a waste of time. Boring plot and terrible acting.")
# [LogisticRegression] NEGATIVE (100%)  <-  What a waste of time. ...

predict_sentiment("Great value and durable build.", model="MultinomialNB")
# returns (label, confidence), e.g. ("positive", 0.7)
```

- `model` can be `"LogisticRegression"` (default) or `"MultinomialNB"`.
- It returns a `(label, confidence)` tuple, where confidence is the predicted-class probability. Pass `verbose=False` to suppress printing.
- For a live input loop, uncomment `interactive_test()` in the notebook. Press Enter on an empty line to exit.

## Limitations

- **Domain shift:** the model is trained on movie reviews, so product-specific vocabulary (e.g. "stopped working", "damaged") transfers poorly. For production use, retrain on in-domain product reviews. Only the data-loading section needs to change, provided the new file has `review_text` and `sentiment` columns.
- **Bag-of-words limits:** sarcasm, long-range negation and mixed sentiment are hard for TF-IDF. For example, "Not bad, actually quite enjoyable." is misclassified as negative.
- **Binary only:** the `neutral` class in the starter file is not modelled.

## Possible extensions

- Hyperparameter search (`GridSearchCV` over `C`, `alpha`, `ngram_range`).
- spaCy lemmatization with POS tags.
- Error analysis on misclassified reviews.
- Three-class (positive / neutral / negative) classification.
