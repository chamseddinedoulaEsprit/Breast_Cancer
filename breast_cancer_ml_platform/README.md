# 🏥 ML Diagnostic Platform - Breast Cancer

## 📋 Description

Intelligent system for malignant tumor detection with medical explainability, developed with Flask. This project implements the **3 academic Business Objectives**:

- **BO-1**: Rapid and accurate detection of malignant/benign tumors
- **BO-2**: Explainability and transparency of predictions
- **BO-3**: Risk stratification for better management

## 🎯 Use Cases

### 👤 Patient
- **UC-P1**: Submit medical data
- **UC-P2**: View diagnostic result
- **UC-P3**: View risk level
- **UC-P4**: Access simplified explanations

### 👨‍⚕️ Doctor
- **UC-D1**: Access patient predictions
- **UC-D2**: Analyze feature importance
- **UC-D3**: Evaluate risk stratification
- **UC-D4**: Compare cases

### 🛠️ Admin
- **UC-A1**: Manage users (CRUD)
- **UC-A2**: Manage ML models
- **UC-A3**: Monitor system performance
- **UC-A4**: Configure explainability settings
- **UC-A5**: Manage security and data

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Installation Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database and run application
python app.py
```

The application will be accessible at: **http://localhost:5000**

## 👥 Demo Accounts

| Role | Email | Password |
|------|-------|--------------|
| Admin | admin@medical.com | admin123 |
| Doctor | doctor@medical.com | doctor123 |
| Patient | Create your own account | - |

## 📊 Medical Features

The model uses **10 features** from the Breast Cancer Wisconsin dataset:

1. **radius_mean**: Mean radius of tumor cells
2. **texture_mean**: Mean texture (variation of gray levels)
3. **perimeter_mean**: Mean perimeter of the tumor
4. **area_mean**: Mean area
5. **smoothness_mean**: Surface smoothness
6. **compactness_mean**: Compactness (perimeter² / area - 1)
7. **concavity_mean**: Mean concavity
8. **concave_points_mean**: Mean number of concave points
9. **symmetry_mean**: Tumor symmetry
10. **fractal_dimension_mean**: Fractal dimension (contour complexity)

## 🧠 ML Model

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~95%
- **Number of estimators**: 100
- **Validation**: Train/Test Split (80/20)

## 🏗️ Architecture

```
breast_cancer_ml_platform/
├── app.py                 # Main Flask Application
├── models.py              # Database models
├── forms.py               # WTForms Forms
├── ml_service.py          # ML Service (predictions, explainability)
├── requirements.txt       # Python Dependencies
├── templates/             # HTML Templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── patient_*.html
│   ├── doctor_*.html
│   └── admin_*.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
├── models/                # Saved ML Models
│   ├── breast_cancer_model.joblib
│   └── scaler.joblib
└── medical_ml.db          # SQLite Database
```

## 🔐 Security

- Password hashing with Werkzeug (pbkdf2:sha256)
- Session-based authentication with Flask-Login
- CSRF protection with Flask-WTF
- Role-based access control

## 📈 Key Features

### For Patients
- ✅ Intuitive data entry interface
- ✅ Real-time prediction
- ✅ Risk level assessment
- ✅ Simplified explanations
- ✅ Prediction history

### For Doctors
- ✅ Patient overview
- ✅ Detailed feature analysis
- ✅ Feature importance with visualizations
- ✅ Risk stratification
- ✅ Clinical recommendations

### For Admins
- ✅ Complete user management
- ✅ Upload and activation of ML models
- ✅ Statistics and performance metrics
- ✅ System monitoring

## 🎨 Technologies Used

- **Backend**: Flask, Flask-Login, Flask-SQLAlchemy
- **Frontend**: Bootstrap 5, Font Awesome, Chart.js
- **ML**: scikit-learn, pandas, numpy, joblib
- **Database**: SQLite
- **Explainability**: Feature importance (Random Forest)

## 📝 Prediction Example

```python
# Example of benign data
input_data = {
    'radius_mean': 12.5,
    'texture_mean': 18.0,
    'perimeter_mean': 80.0,
    'area_mean': 500.0,
    'smoothness_mean': 0.09,
    'compactness_mean': 0.08,
    'concavity_mean': 0.05,
    'concave_points_mean': 0.03,
    'symmetry_mean': 0.17,
    'fractal_dimension_mean': 0.06
}
```

## 🔄 Workflow

1. **Patient** submits medical data via the form
2. The system **preprocesses** the data (normalization)
3. The **ML model** performs the prediction
4. Calculation of **risk level** and **feature importance**
5. Saving to the **database**
6. Displaying **results** with explanations

## 📊 Performance Metrics

- **Accuracy**: 95%
- **Precision**: 93%
- **Recall**: 96%
- **F1-Score**: 94%

## 🆘 Support

For any question or problem:
- Consult the documentation in the code
- Check application logs
- Contact the development team

## 📜 License

Academic Project - University 2025

## 👨‍💻 Author

Developed for a university ML deployment project

---

**⚠️ Warning**: This system is a support tool and does not replace a professional medical diagnosis.
