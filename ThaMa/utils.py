import joblib
import numpy as np
import os
import gdown

# Resource management
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'resources')
GDRIVE_FOLDER_ID = "1eDOVkDPS4pLG7KPvz8eImbUIdVCIrMu4"

def download_resources():
    """Download required resources from Google Drive"""
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    
    # Download folder from Google Drive
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    gdown.download_folder(url=url, output=RESOURCES_DIR, quiet=False)
    
    print(f"Resources downloaded to {RESOURCES_DIR}")

def load_resources():
    """Load required resources or download if not present"""
    if not os.path.exists(RESOURCES_DIR):
        print("Downloading required resources...")
        download_resources()
    
def load_language_resources(language):
    """Load resources for specific language"""
    if language.lower() not in ['tamil', 'malayalam']:
        raise ValueError("Language must be either 'tamil' or 'malayalam'")
        
    # Load the scalers
    global scaler_mt5, scaler_roberta, scaler_indic, scaler_sentence, label_encoder
    
    scaler_mt5 = joblib.load(os.path.join(RESOURCES_DIR, f'utils/{language}/MT5_scaler.pkl'))
    scaler_roberta = joblib.load(os.path.join(RESOURCES_DIR, f'utils/{language}/XLM_scaler.pkl'))
    scaler_indic = joblib.load(os.path.join(RESOURCES_DIR, f'utils/{language}/indic_scaler.pkl'))
    scaler_sentence = joblib.load(os.path.join(RESOURCES_DIR, f'utils/{language}/sentence_scaler.pkl'))
    
    # Load the label encoder
    label_encoder = joblib.load(os.path.join(RESOURCES_DIR, f'utils/{language}/label_encoder.pkl'))

# Initialize resources
load_resources()

def preprocess_input(X, scaler):
    return scaler.transform(X)

def ensemble_predict_weights(model_mt5, model_roberta, model_indic, model_sentence, features, language, weights):

    load_language_resources(language)

    # Scale inputs for each model
    X_mt5, X_roberta, X_indic, X_sentence = features[0], features[1], features[2], features[3]
    X_mt5 = preprocess_input(X_mt5, scaler_mt5)
    X_roberta = preprocess_input(X_roberta, scaler_roberta)
    X_indic = preprocess_input(X_indic, scaler_indic)
    X_sentence = preprocess_input(X_sentence, scaler_sentence)

    # Get predictions (probabilities) from each model
    pred_mt5 = model_mt5.predict(X_mt5, verbose=0)
    pred_roberta = model_roberta.predict(X_roberta, verbose=0)
    pred_indic = model_indic.predict(X_indic, verbose=0)
    pred_sentence = model_sentence.predict(X_sentence, verbose=0)
    
    # Combine predictions using weighted average
    combined_preds = (
        weights[0] * pred_mt5 +
        weights[1] * pred_roberta +
        weights[2] * pred_indic +
        weights[3] * pred_sentence
    )


    # Convert weighted probabilities to class predictions
    final_preds = np.argmax(combined_preds, axis=1)

    return label_encoder.inverse_transform(final_preds)
