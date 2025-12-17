# 📖 Installation and User Guide

## 🚀 Quick Installation

### Method 1: Automatic Script (Recommended)

```bash
# 1. Extract the archive
unzip breast_cancer_ml_platform.zip
cd breast_cancer_ml_platform

# 2. Run the startup script
./start.sh
```

### Method 2: Manual Installation

```bash
# 1. Extract the archive
unzip breast_cancer_ml_platform.zip
cd breast_cancer_ml_platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python run.py
```

## 🌐 Accessing the Application

Once started, access at: **http://localhost:5000**

## 👥 Test Accounts

### Administrator
- **Email**: admin@medical.com
- **Password**: admin123
- **Access**: User management, ML models, statistics

### Doctor
- **Email**: doctor@medical.com  
- **Password**: doctor123
- **Access**: Patient analysis, feature importance, risk stratification

### Patient
- **Create your own account** via the "Register" button
- **Access**: Data submission, result visualization, history

## 🎯 Use Cases by Role

### 👤 Patient

#### UC-P1: Submit Medical Data
1. Log in with your patient account
2. Click on **"New Prediction"**
3. Fill in the 10 medical features
4. Click on **"Get Prediction"**

**Example values (Benign Case)**:
```
radius_mean: 12.5
texture_mean: 18.0
perimeter_mean: 80.0
area_mean: 500.0
smoothness_mean: 0.09
compactness_mean: 0.08
concavity_mean: 0.05
concave_points_mean: 0.03
symmetry_mean: 0.17
fractal_dimension_mean: 0.06
```

#### UC-P2: View Diagnostic Result
- After submission, you will immediately see:
  - ✅ Prediction: **Malignant** or **Benign**
  - 📊 Model confidence level
  - 💬 Simplified interpretation

#### UC-P3: View Risk Level
- The system automatically displays:
  - 🟢 **Low**: Low risk
  - 🟡 **Medium**: Medium risk
  - 🔴 **High**: High risk

#### UC-P4: Access Explanations
- Visualize the **top 5 features** that influenced the prediction
- Progress charts showing the importance of each feature

---

### 👨‍⚕️ Doctor

#### UC-D1: Access Patient Predictions
1. Log in with the doctor account
2. Dashboard displays all recent predictions
3. Click on **"Patients"** to see the full list

#### UC-D2: Analyze Feature Importance
1. On the patient list, click **"Analyze"**
2. You will see:
   - Detailed **feature importance** table
   - Visualizations with progress bars
   - Clinical interpretation of top 3 features

#### UC-D3: Evaluate Risk Stratification
- Each patient analysis includes:
  - **Model confidence score**
  - **Risk level** (Low/Medium/High)
  - **Tailored clinical recommendations**
  - **Clinical priority** (Standard/Medium/High)

#### UC-D4: Compare Cases
- Use the patient list to:
  - Filter by type (Malignant/Benign)
  - Compare risk levels
  - Identify patterns

---

### 🛠️ Admin

#### UC-A1: Manage Users
1. Menu **"Users"**
2. **Create**: "Create User" button
   - Enter name, email, password
   - Choose role (Patient/Doctor/Admin)
3. **Delete**: Red "Trash" button on each row

#### UC-A2: Manage ML Models
1. Menu **"Models"**
2. **Upload**: "Upload Model" button
   - Accepted format: `.joblib`
   - Must have 10 features
3. **Activate**: "Activate" button to switch models

#### UC-A3: Monitor Performance
1. Menu **"Statistics"**
2. Visualize:
   - Total number of predictions
   - Malignant/Benign Distribution
   - Metrics: Accuracy, Precision, Recall, F1-Score
   - Pie chart

---

## 🔧 File Structure

```
breast_cancer_ml_platform/
├── config.py              # Flask and DB Configuration
├── run.py                 # Main entry point
├── app.py                 # Application routes
├── models.py              # Database models
├── forms.py               # WTForms Forms
├── ml_service.py          # ML Service (predictions)
├── templates/             # Jinja2 HTML Templates
├── static/                # CSS, JS, assets
├── models/                # ML Models (.joblib)
├── medical_ml.db          # SQLite Database
└── requirements.txt       # Python Dependencies
```

## 🐛 Troubleshooting

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "Port 5000 already in use"
Modify in `run.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Error: "Database locked"
```bash
rm medical_ml.db
python run.py
```

### ML Model not found
The system will automatically create a default model on first launch.

## 📊 Performance Metrics

The default Random Forest model achieves:
- **Accuracy**: 95%
- **Precision**: 93%
- **Recall**: 96%
- **F1-Score**: 94%

## 🔐 Security

- Passwords: hashed with **pbkdf2:sha256**
- Sessions: managed by **Flask-Login**
- CSRF: protection with **Flask-WTF**
- Access control: role-based

## 📱 Browser Support

✅ Chrome, Firefox, Safari, Edge (latest versions)

## ⚠️ Medical Warning

This system is a **support tool** for research and education.  
**It does NOT replace a professional medical diagnosis.**

---

**🎓 University Project 2025**  
ML Diagnostic Platform with Explainability
