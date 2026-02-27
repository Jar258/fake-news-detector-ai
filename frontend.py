import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("🚨 Fake News Detector v2.0")
st.markdown("**RoBERTa + FastAPI + SHAP Explainability**")

news_text = st.text_area("Paste news article:", height=200)

if st.button("🔍 Analyze News", type="primary"):
    if news_text:
        with st.spinner("RoBERTa analyzing..."):
            response = requests.post(API_URL, json={"content": news_text})
            
        if response.status_code == 200:
            result = response.json()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Verdict:** {result['label']} ({result['confidence']}%)")
                st.markdown(f"**Why:** {result['explainability']['reasoning']}")
            with col2:
                color = "🔴" if result['label'] == "FAKE" else "🟢"
                st.markdown(f"### {color}")
                
        else:
            st.error("API error. Check server.")
    else:
        st.warning("Enter news text")
