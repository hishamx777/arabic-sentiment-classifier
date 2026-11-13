import re
import unicodedata
import joblib
import streamlit as st

# Must be identical to the clean() used for training in the notebook
noise = re.compile(
    "[\U0001F000-\U0001FAFF\u2190-\u21FF\u2300-\u23FF"
    "\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]"
)

def clean(t):
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[\u064B-\u0652\u0640]", "", t)  # diacritics + tatweel
    t = noise.sub("", t)
    return re.sub(r"\s+", " ", t).strip()

@st.cache_resource
def load():
    return joblib.load("baseline.joblib")

vec, clf = load()

st.title("Arabic Tweet Sentiment")
st.caption(
    "TF-IDF + Logistic Regression baseline (the fine-tuned MARBERTv2 model is "
    "too large for free hosting; see the README for its results)."
)

text = st.text_area("اكتب تغريدة", height=120)

if st.button("حلّل"):
    cleaned = clean(text)
    if not cleaned:
        st.warning("النص فاضي بعد التنضيف (الإيموجي بيتشال).")
    else:
        probs = clf.predict_proba(vec.transform([cleaned]))[0]
        best = clf.classes_[probs.argmax()]
        st.subheader(f"{best}  ({probs.max():.0%})")
        for c, p in zip(clf.classes_, probs):
            st.progress(float(p), text=f"{c}: {p:.1%}")

st.info(
    "Labels in the training data were derived from emojis, so the model "
    "reflects tweet style more than true sentiment. See Limitations in the README."
)
