# Arabic Tweet Sentiment Classifier

Binary (negative / positive) sentiment classification of Arabic tweets.
A TF-IDF + Logistic Regression baseline is compared against a fine-tuned **MARBERTv2**.
The project covers the full pipeline: **Data → Model → Evaluation → Deployment**.

| | Link |
|---|---|
| Live demo (baseline, Streamlit) | TODO: Streamlit URL |
| Fine-tuned model | https://huggingface.co/hishamx777/arabic-sentiment-marbert |
| Demo video (fine-tuned model) | TODO: video URL |
| Notebook | [`arabic_sentiment.ipynb`](arabic_sentiment.ipynb) |

## Results

Evaluated once on a held-out test set of 6,225 tweets (3,287 neg / 2,938 pos), using the final pipeline.

| Model | Macro F1 | Accuracy | Recall neg | Recall pos |
|---|---|---|---|---|
| TF-IDF (1-2 grams) + Logistic Regression | 0.666 | 0.67 | 0.74 | 0.59 |
| MARBERTv2, fine-tuned (3 epochs) | **0.730** | 0.74 | 0.87 | 0.59 |

The 6-point gain is larger than the sampling noise of a test set this size (roughly ±1 point); I did not run a formal significance test.
Both models miss positive tweets far more often than negative ones (positive recall 0.59), so most errors are positive tweets predicted negative.

### Run-to-run variation
I trained MARBERTv2 twice with the same seed (GPU/fp16 training is not fully deterministic). Macro F1 was almost identical (0.729 and 0.730), but the class balance of the errors shifted noticeably (positive recall 0.68 in the first run, 0.59 in the second). The published model is the second run. Treat a single run's per-class numbers with caution.

### Text normalization ablation
| Cleaning | Baseline Macro F1 | MARBERTv2 Macro F1 |
|---|---|---|
| Emoji removal only | 0.668 | 0.729 |
| + Unicode NFKC and diacritics/tatweel removal (final) | 0.666 | 0.730 |

No measurable difference (within noise). It is kept in the final pipeline because it is harmless and turns decorative Arabic presentation-form text into standard letters.

## Data

[Arabic Sentiment Twitter Corpus]([https://www.kaggle.com/](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus)) (original source: Kaggle, `mksaad`), loaded from the Hugging Face mirror `asas-ai/Arabic_Sentiment_Twitter_Corpus`.
The mirror's dataset card lists the CDLA-Sharing-1.0 license. <!-- TODO: confirm on the original Kaggle page and fix the Kaggle link above -->
The data is **not** redistributed in this repo; the notebook downloads it.

Raw size: 45,275 train / 11,520 test. After cleaning: 28,768 train / 6,225 test.

### Cleaning decisions (and why)

1. **Emoji removal.** The labels come from distant supervision: a tweet was labeled by the emojis in it, and ~84% of tweets contain emojis. Keeping them lets a model "read" the label from the emoji instead of learning language, which inflates scores.
2. **Duplicates removed after emoji removal.** About a third of the train set were exact duplicates. Doing this *after* removing emojis also catches tweets that differed only by emoji.
3. **Conflicting labels dropped.** Texts that appear with both labels once emojis are gone have no linguistic signal to learn from.
4. **Train/test overlap removed**, so no tweet appears in both splits.
5. A validation set (10% of train) is used for choosing the best epoch; the test set is touched only for the final numbers.

## Error analysis

Overall 1,627 of 6,225 test tweets are misclassified. In a random sample of 25 errors, 15 were positive tweets predicted negative and 10 the reverse. Recurring patterns:

- **Tweets with no real sentiment** whose label came from a removed emoji (greetings, reminders, announcements, questions).
- **Labels that look wrong, with the model arguably right.** Some tweets labeled positive read as negative or insulting, and some labeled negative read as encouraging. This is the cost of emoji-derived labels, and it caps the achievable score.
- **Sarcasm and joking tone**, and dialect expressions (Gulf, Egyptian) the model has little data for.
- **Poetic or religious text** (supplications, quotes), which the model tends to read as sad.
- **Decorative text** using unusual Unicode characters.

<!-- TODO: read the 25 errors yourself, count each category, and put the counts here -->

## Limitations

- Labels are emoji-derived, so the model partly learns tweet style rather than true sentiment; neutral tweets are forced into pos/neg.
- The corpus leans toward Gulf dialect. Performance on Egyptian Arabic has not been measured on a labeled external set.
- The live demo runs the baseline only: the free hosting tier has too little memory for MARBERTv2 (~650 MB of weights). The fine-tuned model is on the Hub and shown in the video.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Reproduce training: open `arabic_sentiment.ipynb` in Colab with a T4 GPU and run all cells (about 8 minutes for fine-tuning).
`app.py` expects `baseline.joblib`, which the notebook produces.

## Repo layout

```
arabic_sentiment.ipynb   data, baseline, fine-tuning, evaluation
app.py                   Streamlit demo (baseline)
baseline.joblib          trained TF-IDF + LogReg
requirements.txt         pinned to the scikit-learn version used for training
```
