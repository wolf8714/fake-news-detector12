import streamlit as st
import joblib
import re

model = joblib.load('news_model.pkl')
vec = joblib.load('news_vectorizer.pkl')

def clean(t):
    t = str(t).lower()
    t = re.sub(r'\(reuters\)', '', t)
    t = re.sub(r'^\s*[a-z\s]+\(reuters\)\s*-', '', t)
    t = re.sub(r'reuters', '', t)
    t = re.sub(r'http\S+', '', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
<style>
.stApp { background-color: #14171C; }
h1 { font-family: Georgia, serif; color: #F5F3EE; }
p, label, .stMarkdown { color: #C9CDD4; }
.verdict { border-radius: 12px; padding: 24px; text-align: center; margin-top: 18px; }
.fake { background: #3A1518; color: #FF6B6B; }
.real { background: #14301F; color: #5FD08A; }
.verdict .v { font-family: Georgia, serif; font-size: 34px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("Fake News Detector")
st.write("Paste a news article below to check whether it looks real or fake.")

text = st.text_area("Article text", height=240, placeholder="Paste the full article here...")

if st.button("Analyse", type="primary"):
    if len(text.strip()) < 50:
        st.warning("Please paste a longer article (at least a few sentences).")
    else:
        v = vec.transform([clean(text)])
        prob = model.predict_proba(v)[0]
        fake_p = prob[1] * 100
        if fake_p >= 50:
            st.markdown(f'<div class="verdict fake"><div class="v">Likely FAKE</div>'
                        f'<div>Confidence: {fake_p:.1f}%</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict real"><div class="v">Likely REAL</div>'
                        f'<div>Confidence: {100-fake_p:.1f}%</div></div>', unsafe_allow_html=True)
        st.caption("Trained on the ISOT dataset. Accuracy reflects source-style patterns and is not a guarantee of truthfulness.")
