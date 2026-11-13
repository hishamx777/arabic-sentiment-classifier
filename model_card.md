---
language: ar
base_model: UBC-NLP/MARBERTv2
datasets:
  - asas-ai/Arabic_Sentiment_Twitter_Corpus
metrics:
  - f1
pipeline_tag: text-classification
---

# arabic-sentiment-marbert

`UBC-NLP/MARBERTv2` fine-tuned for binary (neg / pos) sentiment classification of Arabic tweets.

## Results (held-out test set, 6,225 tweets after cleaning)

| Model | Macro F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.666 |
| This model | 0.730 |

Per class: recall 0.87 (neg) and 0.59 (pos). Most errors are positive tweets predicted negative.

## Training data
Arabic Sentiment Twitter Corpus (original source: Kaggle, mksaad), loaded from the Hugging Face mirror `asas-ai/Arabic_Sentiment_Twitter_Corpus`.
Unicode NFKC normalization, diacritics/tatweel and emojis were removed, duplicates and conflicting labels dropped, and train/test overlap removed.

## Limitations
- The corpus labels were derived from emojis (distant supervision). The model learns tweet style as much as sentiment, and neutral tweets get a forced pos/neg label.
- Mostly Gulf dialect; weaker on Egyptian and other dialects (not yet measured on a labeled external set).
- Sarcasm, jokes and short religious phrases are frequently misclassified. Do not use for moderation or safety decisions.

## Preprocessing (required at inference)
Apply the `clean()` function from the GitHub repo exactly (NFKC normalization, diacritics/tatweel removal, emoji removal, whitespace collapse).

## Licenses
Base model: UBC-NLP/MARBERTv2 (TODO: check the license on its Hub page).
Data: see the dataset page. Code and weights here: TODO choose a license.
