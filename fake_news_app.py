import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 AI Fake News Detector")
st.write("Enter a news article below and check if it's REAL or FAKE")

# Input box
news = st.text_area("✍️ Enter News Text Here")

# Button
if st.button("Check News"):

    if news.strip() == "":
        st.warning("⚠️ Please enter some news text")
    else:
        # Transform input
        data = vectorizer.transform([news])

        # Predict
        prediction = model.predict(data)

        # Output
        if prediction[0] == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Machine Learning")