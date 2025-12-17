
from flask import Flask
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import Email

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

class TestForm(FlaskForm):
    email = StringField('Email', validators=[Email()])

with app.test_request_context('/'):
    form = TestForm(email='test@example.com')
    if form.validate():
        print("Validation successful")
    else:
        print(f"Validation failed: {form.errors}")
