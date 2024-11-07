import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Suicide Ideation Detection", layout="wide")
st.title("Suicide Ideation Detection in Tweets")

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open("best_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.markdown("""
This app classifies tweets into three categories:
- **Suicidal**: Tweets that indicate clear suicidal intent.
- **Potential Suicide Post**: Tweets that show signs of potential suicide ideation.
- **Not Suicidal Post**: Tweets that are not related to suicide ideation.
""")

st.write("### Enter a tweet to classify:")

user_input = st.text_area("Tweet Text", height=150)


def preprocess_text(text):
    text = text.lower()
    return text


def classify_text(text):
    processed_input = preprocess_text(text)
    input_tfidf = tfidf.transform([processed_input])
    prediction = model.predict(input_tfidf)
    prediction_prob = model.predict_proba(input_tfidf).max()
    return processed_input, prediction[0], prediction_prob


def motivational_message_for_suicidal():
    return (
        "Please remember that you are not alone. If you ever feel like you're struggling, "
        "talk to someone you trust or reach out to a professional who can help. "
        "Your well-being matters, and there is support available for you."
    )


def motivational_message_for_potential_suicide():
    return (
        "It's okay to feel overwhelmed sometimes, but remember, help is always available. "
        "Please talk to someone you trust or a mental health professional. You are not alone, and things can get better."
    )


if st.button("Classify Tweet"):
    if user_input:
        # Classify the input tweet and get the prediction
        processed_input, classification, confidence = classify_text(user_input)

        st.write(f"### Classification: {classification}")
        st.write(f"**Confidence Level**: {confidence * 100:.2f}%")

        # Display word cloud
        st.write("### Word Cloud for the Input Tweet:")
        wordcloud = WordCloud(width=800, height=400, max_words=200, background_color="white").generate(processed_input)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

    else:
        st.warning("Please enter some text to classify.")
