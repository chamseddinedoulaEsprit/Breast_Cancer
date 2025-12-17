from flask_login import UserMixin
from datetime import datetime
from config import db

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # patient, doctor, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='patient', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<User {self.email} - {self.role}>'

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)  # Malignant or Benign
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20))  # Low, Medium, High
    feature_importance = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.prediction}>'
    
    def get_risk_color(self):
        """Return color class based on risk level"""
        if self.risk_level == 'Low':
            return 'success'
        elif self.risk_level == 'Medium':
            return 'warning'
        else:
            return 'danger'
    
    def get_prediction_color(self):
        """Return color class based on prediction"""
        if self.prediction == 'Benign':
            return 'success'
        else:
            return 'danger'

class MLModel(db.Model):
    __tablename__ = 'ml_models'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    is_active = db.Column(db.Boolean, default=False)
    accuracy = db.Column(db.Float)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<MLModel {self.name}>'
