import numpy as np
import pandas as pd
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the ML model or create a default one"""
        model_path = 'models/breast_cancer_model.joblib'
        scaler_path = 'models/scaler.joblib'
        features_path = 'models/selected_features.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Load features if available
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
                print(f"✅ Features loaded: {len(self.feature_names)} features")
                
                # Fix for Scaler Mismatch:
                # The scaler was fitted on ALL features (30), but we only use 23.
                # We need to slice the scaler to only include the selected features.
                if hasattr(self.scaler, 'feature_names_in_') and len(self.scaler.feature_names_in_) != len(self.feature_names):
                    print(f"⚠️ Adapting Scaler: {len(self.scaler.feature_names_in_)} -> {len(self.feature_names)} features")
                    
                    try:
                        # Find indices of selected features
                        indices = [list(self.scaler.feature_names_in_).index(f) for f in self.feature_names]
                        
                        # Create a new scaler instance
                        new_scaler = StandardScaler()
                        new_scaler.mean_ = self.scaler.mean_[indices]
                        new_scaler.scale_ = self.scaler.scale_[indices]
                        new_scaler.var_ = self.scaler.var_[indices]
                        new_scaler.n_samples_seen_ = self.scaler.n_samples_seen_
                        new_scaler.n_features_in_ = len(self.feature_names)
                        new_scaler.feature_names_in_ = np.array(self.feature_names)
                        
                        self.scaler = new_scaler
                        print("✅ Scaler adapted successfully!")
                    except Exception as e:
                        print(f"❌ Error adapting scaler: {e}")

            print("✅ Model loaded successfully!")
        else:
            print("⚠️ No model found. Training a new model...")
            self.train_default_model()
    
    def train_default_model(self):
        """Train a default Random Forest model using breast cancer dataset"""
        # Load dataset
        data = load_breast_cancer()
        X = data.data[:, :10]  # Use only first 10 features
        y = data.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        accuracy = self.model.score(X_test_scaled, y_test)
        print(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/breast_cancer_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
    
    def predict(self, input_data):
        """Make prediction on input data"""
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Ensure correct feature order
            df = df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(df)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, [float(x) for x in self.model.feature_importances_]))
            elif hasattr(self.model, 'coef_'):
                # For linear models (Logistic Regression, Linear SVM)
                importances = np.abs(self.model.coef_[0])
                # Normalize to sum to 1
                if np.sum(importances) > 0:
                    importances = importances / np.sum(importances)
                feature_importance = dict(zip(self.feature_names, [float(x) for x in importances]))
            else:
                # Fallback for models without feature_importances_ (like SVM with RBF kernel)
                # We return equal importance or 0
                feature_importance = {name: 1.0/len(self.feature_names) for name in self.feature_names}
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[-1], 
                                           reverse=True))
            
            # Determine prediction label
            prediction_label = 'Benign' if prediction == 1 else 'Malignant'
            confidence = probability[prediction]
            
            # Calculate risk level for benign cases
            risk_level = self.calculate_risk_level(input_data, prediction_label, confidence)
            
            return {
                'prediction': prediction_label,
                'probability': float(confidence),
                'risk_level': risk_level,
                'feature_importance': feature_importance,
                'all_probabilities': {
                    'malignant': float(probability[0]),
                    'benign': float(probability[1])
                }
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_risk_level(self, input_data, prediction, confidence):
        """Calculate risk stratification"""
        if prediction == 'Malignant':
            return 'High'
        
        # For benign cases, calculate risk based on features and confidence
        if confidence >= 0.9:
            return 'Low'
        elif confidence >= 0.7:
            return 'Medium'
        else:
            return 'High'
    
    def get_feature_explanation(self, feature_name, value, importance):
        """Generate human-readable explanation for a feature"""
        explanations = {
            'radius_mean': f"The mean radius of tumor cells is {value:.2f}. This feature has an importance of {importance:.2%} in the prediction.",
            'texture_mean': f"The mean texture (variation of gray levels) is {value:.2f}. Importance: {importance:.2%}.",
            'perimeter_mean': f"The mean perimeter is {value:.2f}. Importance: {importance:.2%}.",
            'area_mean': f"The mean area is {value:.2f}. Importance: {importance:.2%}.",
            'smoothness_mean': f"The mean smoothness (local variation in radius lengths) is {value:.4f}. Importance: {importance:.2%}.",
            'compactness_mean': f"The mean compactness (perimeter^2 / area - 1.0) is {value:.4f}. Importance: {importance:.2%}.",
            'concavity_mean': f"The mean concavity is {value:.4f}. Importance: {importance:.2%}.",
            'concave_points_mean': f"The mean number of concave points is {value:.4f}. Importance: {importance:.2%}.",
            'symmetry_mean': f"The mean symmetry is {value:.4f}. Importance: {importance:.2%}.",
            'fractal_dimension_mean': f"The mean fractal dimension is {value:.4f}. Importance: {importance:.2%}."
        }
        
        return explanations.get(feature_name, f"{feature_name}: {value}")
    
    def get_top_features(self, feature_importance, n=5):
        """Get top N most important features"""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
