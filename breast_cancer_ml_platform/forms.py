from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FloatField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError, NumberRange, Optional

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    name = StringField('Full Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', 
                                    validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField("Register")

class PredictionForm(FlaskForm):
    """Form for patient to submit tumor data"""
    
    # Mean features
    radius_mean = FloatField('Mean Radius', validators=[Optional()], default=14.5, render_kw={"placeholder": "ex: 14.5"})
    texture_mean = FloatField('Mean Texture', validators=[Optional()], default=19.8, render_kw={"placeholder": "ex: 19.8"})
    perimeter_mean = FloatField('Mean Perimeter', validators=[Optional()], default=92.3, render_kw={"placeholder": "ex: 92.3"})
    area_mean = FloatField('Mean Area', validators=[Optional()], default=655.0, render_kw={"placeholder": "ex: 655.0"})
    smoothness_mean = FloatField('Mean Smoothness', validators=[Optional()], default=0.096, render_kw={"placeholder": "ex: 0.096"})
    compactness_mean = FloatField('Mean Compactness', validators=[Optional()], default=0.104, render_kw={"placeholder": "ex: 0.104"})
    concavity_mean = FloatField('Mean Concavity', validators=[Optional()], default=0.088, render_kw={"placeholder": "ex: 0.088"})
    concave_points_mean = FloatField('Mean Concave Points', validators=[Optional()], default=0.048, render_kw={"placeholder": "ex: 0.048"})
    symmetry_mean = FloatField('Mean Symmetry', validators=[Optional()], default=0.181, render_kw={"placeholder": "ex: 0.181"})
    fractal_dimension_mean = FloatField('Mean Fractal Dimension', validators=[Optional()], default=0.062, render_kw={"placeholder": "ex: 0.062"})

    # SE features
    radius_se = FloatField('Radius (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 0.2"})
    texture_se = FloatField('Texture (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 1.0"})
    perimeter_se = FloatField('Perimeter (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 2.0"})
    area_se = FloatField('Area (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 20.0"})
    smoothness_se = FloatField('Smoothness (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 0.005"})
    compactness_se = FloatField('Compactness (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 0.02"})
    concavity_se = FloatField('Concavity (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 0.02"})
    concave_points_se = FloatField('Concave Points (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 0.01"})
    symmetry_se = FloatField('Symmetry (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 0.02"})
    fractal_dimension_se = FloatField('Fractal Dim. (SE)', validators=[Optional()], render_kw={"placeholder": "ex: 0.003"})

    # Worst features
    radius_worst = FloatField('Radius (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 16.0"})
    texture_worst = FloatField('Texture (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 25.0"})
    perimeter_worst = FloatField('Perimeter (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 100.0"})
    area_worst = FloatField('Area (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 800.0"})
    smoothness_worst = FloatField('Smoothness (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 0.12"})
    compactness_worst = FloatField('Compactness (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 0.2"})
    concavity_worst = FloatField('Concavity (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 0.2"})
    concave_points_worst = FloatField('Concave Points (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 0.1"})
    symmetry_worst = FloatField('Symmetry (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 0.3"})
    fractal_dimension_worst = FloatField('Fractal Dim. (Worst)', validators=[Optional()], render_kw={"placeholder": "ex: 0.08"})
    
    submit = SubmitField('Get Prediction')
