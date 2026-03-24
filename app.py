import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, RobertaForSequenceClassification

# ================= CONFIG =================
MODEL_ID = "tanishq-agarkar/MiniProject"  # 🔥 CHANGE THIS

GOEMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]

THRESHOLD = 0.3  # tweak if needed

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ================= UI =================
st.set_page_config(page_title="Emotion Detector", page_icon="🧠")

st.title("🧠 Emotion Detection App")
st.markdown("Detect emotions from text using a fine-tuned RoBERTa model")

# Input box
text = st.text_area("Enter your text:")

# ================= PREDICTION =================
if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        enc = tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Get predictions
        results = [
            (GOEMOTION_LABELS[i], float(p))
            for i, p in enumerate(probs) if p > THRESHOLD
        ]

        # Sort by confidence
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # ================= OUTPUT =================
        if results:
            st.subheader("Detected Emotions:")

            # Show top 5
            for label, score in results[:5]:
                st.write(f"**{label}**: {score:.2f}")

            # Bar chart
            st.subheader("Confidence Scores")
            chart_data = {label: score for label, score in results[:5]}
            st.bar_chart(chart_data)

        else:
            st.info("No strong emotion detected.")

# ================= FOOTER =================
st.markdown("---")
st.markdown("Built with ❤️ using Transformers & Streamlit")
