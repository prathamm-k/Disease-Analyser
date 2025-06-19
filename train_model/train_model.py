import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib
import os
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_model/train_model.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define paths relative to train_model/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    logger.info("Loading datasets...")
    train_path = os.path.join(BASE_DIR, "Training.csv")
    test_path = os.path.join(BASE_DIR, "Testing.csv")
    
    if not os.path.exists(train_path):
        logger.error(f"Training file not found at {train_path}")
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(test_path):
        logger.error(f"Testing file not found at {test_path}")
        raise FileNotFoundError(f"Testing file not found at {test_path}")
    
    try:
        df = pd.read_csv(train_path)
        tr = pd.read_csv(test_path)
        logger.info("Datasets loaded successfully")
        return df, tr
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def get_disease_list(df, tr):
    logger.info("Generating disease list from data...")
    try:
        # Combine unique prognosis values from both datasets
        train_prognosis = df['prognosis'].str.lower().unique()
        test_prognosis = tr['prognosis'].str.lower().unique()
        disease = sorted(list(set(train_prognosis) | set(test_prognosis)))
        logger.info(f"Found {len(disease)} unique diseases: {disease}")
        return disease
    except Exception as e:
        logger.error(f"Error generating disease list: {str(e)}")
        raise

def preprocess_data(df, tr, disease):
    logger.info("Preprocessing data...")
    try:
        # Map prognosis to numeric labels
        def replace_prognosis(data):
            mapping = {disease[i].lower(): i for i in range(len(disease))}
            data['prognosis'] = data['prognosis'].str.lower().map(mapping)
            if data['prognosis'].isna().any():
                logger.warning(f"Found {data['prognosis'].isna().sum()} missing prognosis values; replacing with -1")
                data['prognosis'] = data['prognosis'].fillna(-1).astype(int)
            return data[data['prognosis'] != -1]

        df = replace_prognosis(df)
        tr = replace_prognosis(tr)

        # Get symptom columns (excluding prognosis)
        symptoms = [col for col in df.columns if col != 'prognosis']
        symptoms_cleaned = [symptom for symptom in symptoms if symptom in tr.columns]

        if not symptoms_cleaned:
            logger.error("No common symptoms found between training and testing datasets")
            raise ValueError("No common symptoms found")

        # Prepare features and labels
        X = df[symptoms_cleaned]
        y = df["prognosis"]
        X_test = tr[symptoms_cleaned]
        y_test = tr["prognosis"]

        logger.info(f"Data preprocessed: {len(symptoms_cleaned)} features, {len(df)} training samples, {len(tr)} test samples")
        return X, y, X_test, y_test, symptoms_cleaned
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def train_model(X, y, X_test, y_test, disease):
    logger.info("Training Random Forest Classifier...")
    try:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Perform cross-validation
        logger.info("Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        logger.info(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train on full training data
        clf.fit(X, y)

        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test Set Accuracy: {accuracy:.4f}")

        # Generate classification report with dynamic labels
        labels = list(range(len(disease)))
        report = classification_report(y_test, y_pred, labels=labels, target_names=disease, zero_division=0)
        logger.info("\nClassification Report:\n" + report)

        return clf, accuracy
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def save_artifacts(clf, symptoms_cleaned, disease):
    logger.info("Saving model and artifacts...")
    try:
        joblib.dump(clf, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
        joblib.dump(symptoms_cleaned, os.path.join(MODELS_DIR, "symptoms.pkl"))
        joblib.dump(disease, os.path.join(MODELS_DIR, "diseases.pkl"))
        logger.info("Artifacts saved successfully to %s", MODELS_DIR)
    except Exception as e:
        logger.error(f"Error saving artifacts: {str(e)}")
        raise

def main():
    logger.info("Script started")
    
    try:
        # Load data
        df, tr = load_data()
        
        # Generate disease list
        disease = get_disease_list(df, tr)
        
        # Preprocess data
        X, y, X_test, y_test, symptoms_cleaned = preprocess_data(df, tr, disease)

        # Train model
        clf, accuracy = train_model(X, y, X_test, y_test, disease)

        # Save artifacts
        save_artifacts(clf, symptoms_cleaned, disease)
        
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()