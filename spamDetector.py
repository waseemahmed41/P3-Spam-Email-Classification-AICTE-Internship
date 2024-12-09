import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import os

# Load the saved model and vectorizer using pickle
model_file = 'spam123.pkl'  # Adjust if necessary

# Check if file exists
if os.path.exists(model_file):
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        st.write("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
else:
    st.error(f"Model file not found: {model_file}")

vectorizer_file = 'vec123.pkl'  # Adjust if necessary

# Check if file exists
if os.path.exists(vectorizer_file):
    try:
        with open(vectorizer_file, 'rb') as f:
            cv = pickle.load(f)
        st.write("Vectorizer loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the vectorizer: {e}")
else:
    st.error(f"Vectorizer file not found: {vectorizer_file}")


# Streamlit app layout
st.title("Spam Email Detection")
st.write("Enter an email message to classify it as spam or ham.")

# Input from user
user_input = st.text_area("Email Message", "")

# Button to classify the input
if st.button("Classify"):
    if user_input:
        data = [user_input]
        print(data)
        # Transform the input using the same vectorizer used during training
        email_vectorized = cv.transform([data]).toarray()
        
        # Predict using the trained model
        result = model.predict(email_vectorized)

        # Display the result
        if result[0] == 0:
            st.write("The email is **Spam**.")
        else:
            st.write("The email is **Ham**.")
    else:
        st.write("Please enter an email message to classify.")
