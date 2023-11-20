# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 21:22:30 2023

@author: USER
"""

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

st.title("Email Phishing Classification")

st.markdown(
    """
    <p style="text-align:justify;">
    Technological developments that occur in the current era cause information to be obtained from anywhere easily and efficiently. With the development and easy access to information, a system is needed that can secure data and also our privacy. This is because with the ease of getting information, there can also be cyber crimes that threaten our digital privacy via the internet. One example of cybercrime that can occur is phishing. Phishing is a way of exploiting Internet users to obtain important and sensitive information from these users, which can be used irresponsibly. One of the phishing distributions that can occur is through email, which, in the process of spreading email using attachments or dangerous links sent via email which if clicked can steal data from the user. Therefore, it is important to carry out the process of sorting emails that are phishing so that it does not cause harm to ourselves both in terms of privacy and financially.
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:justify;">
    The dataset that used for building classification model is an csv dataset, which consists of a total of 18650 data and has 2 main classes, namely safe email and phishing email. With data distribution, 11322 safe email data and 7328 phishing email data.
    """,
    unsafe_allow_html=True
)

img4 = Image.open('test.png')
st.image(img4, caption='Dataset Visualization', use_column_width=True)

st.markdown(
    """
    <p style="text-align:justify;">
    The classification process will use the Machine learning algorithm method, namely the Support Vector Classification. The purpose of using the SVC algorithm is because this algorithm is an implementation of the SVM algorithm which is very good for processing data and this SVC algorithm is specifically used for the classification process.
    """,
    unsafe_allow_html=True
)

img1 = Image.open('GUI.png')
st.image(img1, caption='SVC Concept', use_column_width=True)

st.title("Classification Process Using SVC")

# Load the saved model and fitted vectorizer
loaded_model = joblib.load('svm_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Streamlit UI for user input
user_input = st.text_input("Enter a sample input:")

# Process the user input and convert it to TF-IDF features
user_input_processed = vectorizer.transform([user_input])

# Make predictions using the loaded model
prediction = loaded_model.predict(user_input_processed)

# Display the prediction
if user_input:  # Check if there is any user input
    if prediction[0] == 0:
        st.success("Prediction: Safe")
    elif prediction[0] == 1:
        st.error("Prediction: Phishing")
else:
    st.info("No input. Please enter a sample input for classification.")


