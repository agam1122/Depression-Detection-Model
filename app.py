import streamlit as st
import joblib
import re
import pandas as pd

# Setup
st.set_page_config(page_title="Depression Detector", page_icon="🧠")
st.title("🧠 Reddit Based Depression Predictor")
st.markdown("Enter a post body below to see which community it belongs to.")

# Load Model & Vectorizer
@st.cache_resource # This keeps the model in RAM so it doesn't reload on every click
def load_assets():
    model = joblib.load('DD_model_lr.pkl')
    vectorizer = joblib.load('Vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# UI
user_input = st.text_area("Post Content:", placeholder="Type something deep or emotional...")

if st.button("Predict Category"):
    if user_input.strip():
        # Process
        cleaned = clean_text(user_input)
        vec_text = vectorizer.transform([cleaned])
        
        prediction = model.predict(vec_text)[0]
        probs = model.predict_proba(vec_text)[0]
        chart_labels = ["Healthy", "Depressed"]
        chart_data = pd.Series(probs, index=chart_labels)
        
        
        # Display Results
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0.0:
                prediction = "Heyy!! You are Fine"
            else:
                prediction = "Seems depressed"
            st.metric("Predicted Class", f"{prediction}")
        
        with col2:
            confidence = max(probs) * 100
            st.metric("Confidence", f"{confidence:.2f}%")
            
        # Show a bar chart of all probabilities
        st.bar_chart(chart_data)
    else:
        st.warning("Please enter some text first!")