"""
Main Flask Application for Wound Care AI
"""
from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
from config import settings
import os

# Initialize extensions
jwt = JWTManager()
socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)

    # Load configuration from settings object
    app.config['SECRET_KEY'] = settings.SECRET_KEY
    app.config['JWT_SECRET_KEY'] = settings.SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False  # Token không hết hạn (hoặc set timedelta)
    app.config['SQLALCHEMY_DATABASE_URI'] = settings.DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ECHO'] = False
    app.config['DEBUG'] = False
    
    # Session config for OAuth
    app.config['SESSION_COOKIE_NAME'] = 'wound_care_session'
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_SECURE'] = False  # Set True in production with HTTPS

    # Initialize extensions
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    jwt.init_app(app)
    socketio.init_app(app)
    
    # Initialize database
    from models import db
    db.init_app(app)
    
    # Initialize OAuth
    from routes.auth import oauth
    oauth.init_app(app)
    
    # Register blueprints
    from routes.auth import auth_bp
    from routes.analysis import analysis_bp
    from routes.chat import chat_bp
    from routes.pdf_export import pdf_bp
    from routes.doctor_extended import doctor_ext_bp
    from routes.admin import admin_bp
    from routes.doctors import doctors_bp
    from routes.patients import patients_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(pdf_bp)
    app.register_blueprint(doctor_ext_bp)
    app.register_blueprint(doctors_bp)
    app.register_blueprint(patients_bp)
    app.register_blueprint(admin_bp)
    
    # Health check endpoint
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'message': 'Wound Care AI API is running'
        }), 200
    
    # Serve uploaded files
    @app.route('/uploads/<path:filename>')
    def serve_upload(filename):
        from flask import send_from_directory
        return send_from_directory('uploads', filename)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

# Create app instance for Gunicorn
app = create_app()

if __name__ == '__main__':
    # Development server
    socketio.run(app, 
                host=os.getenv('HOST', '0.0.0.0'),
                port=int(os.getenv('PORT', 5001)),
                debug=False)

