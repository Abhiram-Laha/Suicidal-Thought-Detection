import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Suicide Ideation Detection", layout="wide")
st.title("Suicide Ideation Detection in Tweets")

model = pickle.load(open("best_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.markdown("""
This app classifies tweets into three categories:
- **Suicidal**: Tweets that indicate clear suicidal intent.
- **Potential Suicide Post**: Tweets that show signs of potential suicide ideation.
- **Not Suicidal Post**: Tweets that are not related to suicide ideation.
""")

st.write("### Enter a tweet to classify it:")

user_input = st.text_area("Tweet Text", height=150)

def preprocess_text(text):
    text = text.lower()
    return text

def process_file(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    return text

file_uploaded = st.file_uploader("Or upload a text file", type=["txt"])

# Default Streamlit button
if st.button("Classify", key="classify_button", help="Click to classify the tweet or file"):
    if user_input:
        processed_input = preprocess_text(user_input)
        input_tfidf = tfidf.transform([processed_input])
        prediction = model.predict(input_tfidf)
        prediction_prob = model.predict_proba(input_tfidf).max()

        st.write(f"### Classification: {prediction[0]}")
        st.write(f"**Confidence Level**: {prediction_prob * 100:.2f}%")

        st.write("### Word Cloud for the Input Tweet:")
        wordcloud = WordCloud(width=800, height=400, max_words=200, background_color="white").generate(processed_input)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

    elif file_uploaded is not None:
        file_text = process_file(file_uploaded)
        processed_input = preprocess_text(file_text)
        input_tfidf = tfidf.transform([processed_input])
        prediction = model.predict(input_tfidf)
        prediction_prob = model.predict_proba(input_tfidf).max()

        st.write(f"### Classification: {prediction[0]}")
        st.write(f"**Confidence Level**: {prediction_prob * 100:.2f}%")

        st.write("### Word Cloud for the Uploaded File:")
        wordcloud = WordCloud(width=800, height=400, max_words=200, background_color="white").generate(processed_input)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

    else:
        st.warning("Please enter some text or upload a text file to classify.")
