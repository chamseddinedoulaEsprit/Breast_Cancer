
---

# 🏥 ML Diagnostic Platform – Breast Cancer

## 📄 Description

This project aims to improve the early detection of breast cancer using **Machine Learning (ML)** and **Deep Learning (DL)** techniques. It provides a complete solution ranging from exploratory data analysis to an interactive web application designed for patients, doctors, and administrators.

Traditional methods (such as biopsies) can be slow and costly. This platform enables rapid analysis of cellular characteristics obtained through the **FNA (Fine Needle Aspiration)** method to deliver reliable diagnostic support.

## 📂 Project Structure

The repository is organized into two main parts:

### 1. Analysis and Modeling (Root Directory)

* **`BreastCancer.ipynb`**: Jupyter notebook containing exploratory data analysis (EDA), data preprocessing, model training, and performance evaluation.
* **`data.csv`**: The medical dataset used for training.
* **`saved_model/`**: Stores trained models (e.g., Keras, Joblib) ready for deployment.

### 2. Web Application (`breast_cancer_ml_platform/`)

A complete **Flask** application that deploys the models for real-world usage.

* **`app.py`**: Entry point of the web application.
* **`templates/`**: User interfaces (Admin, Doctor, and Patient dashboards).
* **`models/`**: Models specifically used by the web application.

## 🎯 Objectives and Features

The project implements **three business objectives (BOs)**:

* **BO-1: Rapid Detection** – Automatic prediction (Malignant / Benign) with high accuracy.
* **BO-2: Explainability** – Prediction transparency to support medical decision-making (Feature Importance).
* **BO-3: Risk Stratification** – Classification of risk levels (Low, Medium, High) to prioritize patient care.

### User Roles

* **👤 Patient**: Submit medical data, view a simplified diagnosis and risk level.
* **👨‍⚕️ Doctor**: Access detailed predictions, analyze clinical feature importance, and manage patient records.
* **🛠️ Admin**: Manage users, supervise ML models, and ensure system security.

## 🚀 Installation and Getting Started

### Prerequisites

* Python 3.8+
* pip

### Running the Web Application

1. Navigate to the platform directory:

   ```bash
   cd breast_cancer_ml_platform
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the server:

   ```bash
   python app.py
   ```

   The application will be available at: [http://localhost:5000](http://localhost:5000)

### Exploring the Notebooks

To explore the data analysis and training process, open `BreastCancer.ipynb` directly in VS Code or Jupyter.

## 👥 Authors

* **Doula Chamseddine**

---
