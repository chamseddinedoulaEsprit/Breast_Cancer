from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import ast
import numpy as np

from config import app, db
from models import User, Prediction, MLModel
from ml_service import MLService
from forms import LoginForm, RegisterForm, PredictionForm

# Initialize ML Service
ml_service = MLService()

# ============= PUBLIC ROUTES =============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    else:
        if form.is_submitted():
            print(f"Login Form Errors: {form.errors}")
    
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('This email is already in use.', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(
            name=form.name.data,
            email=form.email.data,
            password=hashed_password,
            role='patient'
        )
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! You can now login.', 'success')
        return redirect(url_for('login'))
    else:
        if form.is_submitted():
            print(f"Register Form Errors: {form.errors}")
    
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

# ============= DASHBOARD =============
@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'patient':
        predictions = Prediction.query.filter_by(patient_id=current_user.id).order_by(Prediction.created_at.desc()).limit(5).all()
        return render_template('patient_dashboard.html', predictions=predictions)
    elif current_user.role == 'doctor':
        predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(20).all()
        return render_template('doctor_dashboard.html', predictions=predictions)
    elif current_user.role == 'admin':
        total_users = User.query.count()
        total_predictions = Prediction.query.count()
        active_models = MLModel.query.filter_by(is_active=True).count()
        recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
        
        # Get active model name
        active_model_obj = MLModel.query.filter_by(is_active=True).first()
        active_model_name = active_model_obj.name if active_model_obj else "Random Forest"

        return render_template('admin_dashboard.html', 
                             total_users=total_users,
                             total_predictions=total_predictions,
                             active_models=active_models,
                             recent_predictions=recent_predictions,
                             active_model_name=active_model_name)

# ============= PATIENT ROUTES =============
@app.route('/patient/predict', methods=['GET', 'POST'])
@login_required
def patient_predict():
    if current_user.role != 'patient':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    form = PredictionForm()
    
    # Dynamically set required fields based on active model
    # This is a bit tricky with WTForms, but we can just ignore validation for unused fields
    # or rely on the fact that we only render used fields in the template (if we were doing client-side validation)
    # For server-side, we might need to be careful.
    # Ideally, we should construct the form dynamically, but for now, let's assume the user fills what is shown.
    
    if form.validate_on_submit():
        input_data = {}
        missing_fields = []
        
        # Only extract features required by the active model
        for feature in ml_service.feature_names:
            if hasattr(form, feature):
                val = getattr(form, feature).data
                if val is None:
                    missing_fields.append(feature)
                else:
                    input_data[feature] = val
            else:
                missing_fields.append(feature)
        
        if missing_fields:
             flash(f"Please fill in all required fields: {', '.join(missing_fields)}", 'danger')
             return render_template('patient_predict.html', form=form, feature_names=ml_service.feature_names)

        result = ml_service.predict(input_data)
        
        if 'error' in result:
            flash(f"Error: {result['error']}", 'danger')
            return redirect(url_for('patient_predict'))
        
        prediction = Prediction(
            patient_id=current_user.id,
            input_data=str(input_data),
            prediction=result['prediction'],
            probability=result['probability'],
            risk_level=result.get('risk_level', 'N/A'),
            feature_importance=str(result.get('feature_importance', {}))
        )
        db.session.add(prediction)
        db.session.commit()
        
        flash('Prediction successful!', 'success')
        return redirect(url_for('patient_view_result', prediction_id=prediction.id))
    
    return render_template('patient_predict.html', form=form, feature_names=ml_service.feature_names)

@app.route('/patient/result/<int:prediction_id>')
@login_required
def patient_view_result(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    
    if current_user.role == 'patient' and prediction.patient_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    feature_importance = {}
    if prediction.feature_importance:
        try:
            feature_importance = ast.literal_eval(prediction.feature_importance)
        except:
            try:
                feature_importance = eval(prediction.feature_importance, {"np": np})
            except:
                feature_importance = {}
            
    return render_template('patient_result.html', prediction=prediction, feature_importance=feature_importance)

@app.route('/patient/history')
@login_required
def patient_history():
    if current_user.role != 'patient':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    predictions = Prediction.query.filter_by(patient_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('patient_history.html', predictions=predictions)

# ============= DOCTOR ROUTES =============
@app.route('/doctor/patients')
@login_required
def doctor_patients():
    if current_user.role != 'doctor':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template('doctor_patients.html', predictions=predictions)

@app.route('/doctor/analysis/<int:prediction_id>')
@login_required
def doctor_analysis(prediction_id):
    if current_user.role != 'doctor':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    prediction = Prediction.query.get_or_404(prediction_id)
    patient = db.session.get(User, prediction.patient_id)
    
    feature_importance = {}
    if prediction.feature_importance:
        try:
            # Try safe eval first
            feature_importance = ast.literal_eval(prediction.feature_importance)
        except:
            try:
                # Fallback to eval with numpy context
                feature_importance = eval(prediction.feature_importance, {"np": np})
            except:
                feature_importance = {}
    
    return render_template('doctor_analysis.html', 
                         prediction=prediction, 
                         patient=patient,
                         feature_importance=feature_importance)

# ============= ADMIN ROUTES =============
@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    users = User.query.all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/user/create', methods=['POST'])
@login_required
def admin_create_user():
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.json
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    
    new_user = User(
        name=data['name'],
        email=data['email'],
        password=hashed_password,
        role=data['role']
    )
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': 'User created successfully', 'user_id': new_user.id})

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({'error': 'You cannot delete your own account'}), 400
    
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({'message': 'User deleted successfully'})

@app.route('/admin/models')
@login_required
def admin_models():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    models = MLModel.query.all()
    
    # Get active model
    active_model = MLModel.query.filter_by(is_active=True).first()
    active_model_name = active_model.name if active_model else "Random Forest Classifier"
    
    return render_template('admin_models.html', models=models, active_model_name=active_model_name, active_model=active_model)

@app.route('/admin/model/upload', methods=['POST'])
@login_required
def admin_upload_model():
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    if 'model_file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['model_file']
    model_name = request.form['model_name']
    
    filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    
    new_model = MLModel(
        name=model_name,
        filename=filename,
        is_active=False
    )
    db.session.add(new_model)
    db.session.commit()
    
    return jsonify({'message': 'Model uploaded successfully', 'model_id': new_model.id})

@app.route('/admin/model/<int:model_id>/activate', methods=['POST'])
@login_required
def admin_activate_model(model_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    MLModel.query.update({MLModel.is_active: False})
    
    model = MLModel.query.get_or_404(model_id)
    model.is_active = True
    db.session.commit()
    
    ml_service.load_model()
    
    return jsonify({'message': 'Model activated successfully'})

@app.route('/admin/statistics')
@login_required
def admin_statistics():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    total_predictions = Prediction.query.count()
    malignant_count = Prediction.query.filter_by(prediction='Malignant').count()
    benign_count = Prediction.query.filter_by(prediction='Benign').count()
    
    # Get active model
    active_model = MLModel.query.filter_by(is_active=True).first()
    active_model_name = active_model.name if active_model else "Random Forest Classifier"

    stats = {
        'total_predictions': total_predictions,
        'malignant_count': malignant_count,
        'benign_count': benign_count,
        'accuracy': 0.99,
        'active_model': active_model_name
    }
    
    return render_template('admin_statistics.html', stats=stats)
