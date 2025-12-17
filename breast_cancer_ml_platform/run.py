from config import app, db, login_manager
from werkzeug.security import generate_password_hash
from models import User
import app as application_routes  # Import all routes

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def init_db():
    """Initialize database with default users"""
    with app.app_context():
        db.create_all()
        
        # Create default admin if not exists
        admin = User.query.filter_by(email='admin@medical.com').first()
        if not admin:
            admin_user = User(
                name='Admin',
                email='admin@medical.com',
                password=generate_password_hash('admin123', method='pbkdf2:sha256'),
                role='admin'
            )
            db.session.add(admin_user)
        
        # Create default doctor
        doctor = User.query.filter_by(email='doctor@medical.com').first()
        if not doctor:
            doctor_user = User(
                name='Dr. Smith',
                email='doctor@medical.com',
                password=generate_password_hash('doctor123', method='pbkdf2:sha256'),
                role='doctor'
            )
            db.session.add(doctor_user)
        
        db.session.commit()
        print("✅ Base de données initialisée!")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
