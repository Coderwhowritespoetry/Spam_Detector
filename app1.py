# app.py

import streamlit as st
import joblib as jb 

# Load the trained model and vectorizer
model = jb.load(r"C:\sample of site\machine learning\spam_classifier.pkl")
vectorizer = jb.load(r"C:\sample of site\machine learning\vectorizer.pkl")

# UI Title
st.title("📩 SMS Spam Detector")
st.write("Enter a message and check whether it's Spam or Not.")

# User input box
user_input = st.text_area("Enter your message here:")

# Predict button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # Transform the input using the vectorizer
        new_vec = vectorizer.transform([user_input])
        prediction = model.predict(new_vec)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This message is Spam.")
        else:
            st.success("✅ This message is Not Spam.")
