import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pandas as pd
from .utils import ensemble_predict_weights
from transformers import AutoTokenizer, AutoModel, MT5Tokenizer, MT5Model
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class TextDetector:
    SUPPORTED_LANGUAGES = ['Tamil', 'Malayalam']
    
    def __init__(self, language='Tamil'):
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language must be one of {self.SUPPORTED_LANGUAGES}")
            
        self.language = language
        
        # Initialize models and tokenizers
        try:
            self.mt5_tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
            self.mt5_model = MT5Model.from_pretrained('google/mt5-small')
            self.xlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
            self.xlm_model = AutoModel.from_pretrained('xlm-roberta-base')
            self.indic_model = AutoModel.from_pretrained('ai4bharat/IndicBERTv2-MLM-only')
            self.indic_tokenizer = AutoTokenizer.from_pretrained('ai4bharat/IndicBERTv2-MLM-only')  
            self.sentence_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained models: {str(e)}")

        # Load DNN models
        self.models = []
        model_dir = os.path.join(os.path.dirname(__file__), 'resources', 'models', self.language)
        
        model_files = [
            'MT5_dnn_model.h5', 
            'XLM_dnn_model.h5', 
            'indic_dnn_model.h5',
            'sentence_dnn_model.h5',
        ]
        
        try:
            for model_file in model_files:
                model_path = os.path.join(model_dir, model_file)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                model = load_model(model_path)
                self.models.append(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load DNN models: {str(e)}")

    def predict(self, text):
        """
        Make predictions on input text
        Args:
            text (str): Input text to classify
        Returns:
            float: Probability of text being AI-generated
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        best_weights = {
            'Tamil': [0.0, 0.1, 0.5, 0.4],
            'Malayalam': [0.2, 0.4, 0.4, 0.0]
        }[self.language]

        print("Language:", self.language)
        print("Best weights:", best_weights)
        features = self._extract_features(text)
        prediction = ensemble_predict_weights(
            self.models[0], self.models[1], 
            self.models[2], self.models[3], 
            features, 
            self.language,
            best_weights
        )
        return prediction

    def _extract_features(self, text):
        """
        Extract features from text using all models
        Args:
            text (str): Input text
        Returns:
            list[pd.DataFrame]: List of feature DataFrames from each model
        """
        features = []
        
        # Extract MT5 features
        with torch.no_grad():
            mt5_inputs = self.mt5_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            mt5_outputs = self.mt5_model.encoder(**mt5_inputs)
            mt5_embeddings = mt5_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings_df = pd.DataFrame(mt5_embeddings, columns=[f'embedding_{i}' for i in range(mt5_embeddings.shape[1])])
            features.append(embeddings_df)

        # Extract XLM features
        with torch.no_grad():
            xlm_inputs = self.xlm_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            xlm_outputs = self.xlm_model(**xlm_inputs)
            xlm_embeddings = xlm_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings_df = pd.DataFrame(xlm_embeddings, columns=[f'embedding_{i}' for i in range(xlm_embeddings.shape[1])])
            features.append(embeddings_df)

        # Extract IndicBERT features
        with torch.no_grad():
            indic_inputs = self.indic_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            indic_outputs = self.indic_model(**indic_inputs)
            indic_embeddings = indic_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings_df = pd.DataFrame(indic_embeddings, columns=[f'embedding_{i}' for i in range(indic_embeddings.shape[1])])
            features.append(embeddings_df)

        # Extract sentence embeddings
        sentence_embeddings = self.sentence_model.encode([text])
        embeddings_df = pd.DataFrame(sentence_embeddings, columns=[f'embedding_{i}' for i in range(sentence_embeddings.shape[1])])
        features.append(embeddings_df)

        return features

    @classmethod
    def from_pretrained(cls, language='Tamil'):
        """
        Load a pretrained model for specified language
        Args:
            language: 'Tamil' or 'Malayalam'
        """
        return cls(language=language)