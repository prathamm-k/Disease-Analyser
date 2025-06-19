import streamlit as st
import numpy as np
import joblib
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'train_model', 'models')

# Cache artifact loading
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    if not os.path.exists(model_path):
        st.error("Model file not found. Please run train_model.py first.")
        return None, None, None
    try:
        clf = joblib.load(model_path)
        symptoms = joblib.load(os.path.join(MODELS_DIR, "symptoms.pkl"))
        diseases = joblib.load(os.path.join(MODELS_DIR, "diseases.pkl"))
        return clf, symptoms, diseases
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        return None, None, None

# Load artifacts
clf, symptoms, diseases = load_artifacts()

# Streamlit app
st.title("Disease Analyser using Machine Learning")
st.markdown("A project by Pratham Kairamkonda")

if clf is None:
    st.stop()

# Input section
with st.expander("Enter Symptoms"):
    selected_symptoms = st.multiselect(
        "Select Symptoms (1 or more)",
        options=symptoms,
        default=[],
        help="Choose the symptoms you are experiencing."
    )

# Predict
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        try:
            # Create input vector
            input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
            
            # Predict and get probabilities
            probs = clf.predict_proba([input_vector])[0]
            top_indices = np.argsort(probs)[-2:][::-1]  # Top 2 predictions
            
            # Apply confidence threshold (10%)
            threshold = 0.10
            confident_predictions = [(idx, probs[idx]) for idx in top_indices if probs[idx] >= threshold]
            
            # Display results
            st.subheader("Prediction Results:")
            if not confident_predictions:
                st.warning("No confident predictions available. Try selecting different symptoms or consult a healthcare professional.")
            else:
                for idx, prob in confident_predictions:
                    st.write(f"{diseases[idx].title()}: {prob*100:.2f}% confidence")
                if len(confident_predictions) == 1:
                    st.info("Only one disease met the confidence threshold.")
            
            # Disclaimer
            st.markdown("**Disclaimer**: This is a predictive tool and not a medical diagnosis. Consult a healthcare professional for accurate diagnosis.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# About section
with st.expander("About the Model"):
    st.write(f"This app uses a Random Forest Classifier trained on a dataset of {len(symptoms)} symptoms and {len(diseases)} diseases. The model predicts the most likely disease(s) based on the symptoms you select from 132 symptoms and it shows the most probable disease among the 41 diseases, showing up to two diseases with confidence scores above 10%, because some symptoms are common to multiple diseases and calculates probability for 100% for all diseases.")