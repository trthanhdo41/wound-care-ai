"""
Authentication routes
"""
from flask import Blueprint, request, jsonify, redirect, session
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from models import db, User, Patient, Doctor
from datetime import datetime
from authlib.integrations.flask_client import OAuth
from config import CLIENT_ID, CLIENT_SECRET, FE_URL
from urllib.parse import quote

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# Initialize OAuth
oauth = OAuth()
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    client_kwargs={
        'scope': 'openid email profile'
    }
)

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'password', 'full_name', 'role']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create user
        user = User(
            email=data['email'],
            full_name=data['full_name'],
            role=data['role'],
            phone=data.get('phone'),
            avatar_url=data.get('avatar_url')
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.flush()  # Get user.id
        
        # Create role-specific profile
        if data['role'] == 'patient':
            patient = Patient(
                user_id=user.id,
                date_of_birth=datetime.strptime(data.get('date_of_birth'), '%Y-%m-%d').date() if data.get('date_of_birth') else None,
                gender=data.get('gender'),
                address=data.get('address'),
                diabetes_type=data.get('diabetes_type')
            )
            db.session.add(patient)
        elif data['role'] == 'doctor':
            doctor = Doctor(
                user_id=user.id,
                specialization=data.get('specialization'),
                license_number=data.get('license_number'),
                years_of_experience=data.get('years_of_experience'),
                hospital_affiliation=data.get('hospital_affiliation')
            )
            db.session.add(doctor)
        
        db.session.commit()
        
        # Generate tokens
        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        
        # Validate
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password required'}), 400
        
        # Find user
        user = User.query.filter_by(email=data['email']).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is inactive'}), 403
        
        # Generate tokens
        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token
        }), 200
        
    except Exception as e:
        import traceback
        print(f"❌ Login error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user info"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user.to_dict()
        
        # Add role-specific data
        if user.role == 'patient' and user.patient:
            user_data['patient_profile'] = user.patient.to_dict()
        elif user.role == 'doctor' and user.doctor:
            user_data['doctor_profile'] = user.doctor.to_dict()
        
        return jsonify(user_data), 200
        
    except Exception as e:
        import traceback
        print(f"❌ /me error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    try:
        user_id = get_jwt_identity()
        access_token = create_access_token(identity=str(user_id))
        
        return jsonify({'access_token': access_token}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Update user fields
        if 'full_name' in data:
            user.full_name = data['full_name']
        if 'email' in data:
            # Check if email is already taken by another user
            existing = User.query.filter_by(email=data['email']).first()
            if existing and existing.id != user_id:
                return jsonify({'error': 'Email already in use'}), 400
            user.email = data['email']
        if 'phone' in data:
            user.phone = data['phone']
        if 'avatar_url' in data:
            user.avatar_url = data['avatar_url']
        
        # Update patient-specific fields
        if user.role == 'patient' and user.patient:
            if 'date_of_birth' in data and data['date_of_birth']:
                user.patient.date_of_birth = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date()
            if 'gender' in data:
                user.patient.gender = data['gender']
            if 'address' in data:
                user.patient.address = data['address']
            if 'diabetes_type' in data:
                user.patient.diabetes_type = data['diabetes_type']
        
        db.session.commit()
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500



@auth_bp.route('/login_by_google')
def login_by_google():
    """Initiate Google OAuth login"""
    try:
        redirect_uri = request.url_root.rstrip('/') + '/api/auth/callback'
        print(f"🔗 Google OAuth redirect URI: {redirect_uri}")
        
        # Validate OAuth configuration
        if not CLIENT_ID or not CLIENT_SECRET:
            return jsonify({'error': 'Google OAuth not configured'}), 500
        
        return oauth.google.authorize_redirect(redirect_uri)
    except Exception as e:
        print(f"❌ Error in Google OAuth initiation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to initiate Google OAuth login'}), 500


@auth_bp.route('/callback')
def google_callback():
    """Handle Google OAuth callback"""
    try:
        print("🔄 Processing Google OAuth callback...")
        
        # Get access token from Google
        try:
            token = oauth.google.authorize_access_token()
            print("✅ Successfully obtained access token from Google")
        except Exception as error:
            print(f"❌ OAuth error: {error}")
            error_url = f"{FE_URL}/login?error=oauth_error"
            return redirect(error_url)

        # Get user info from Google
        userinfo = token.get("userinfo")
        if not userinfo:
            print("❌ Failed to retrieve user info from Google")
            error_url = f"{FE_URL}/login?error=no_userinfo"
            return redirect(error_url)

        print(f"👤 Google user info: {userinfo.get('email')}")

        # Normalize user info
        email = userinfo.get("email")
        name = userinfo.get("name", "").replace(" ", "")
        picture = userinfo.get("picture")

        if not email:
            error_url = f"{FE_URL}/login?error=no_email"
            return redirect(error_url)

        # Check if user exists
        user = User.query.filter_by(email=email).first()
        
        if not user:
            # Create new user
            user = User(
                email=email,
                full_name=name,
                avatar_url=picture,
                role='patient'  # Default role
            )
            user.set_password('google_oauth_' + email)  # Dummy password
            db.session.add(user)
            db.session.flush()
            
            # Create patient profile
            patient = Patient(user_id=user.id)
            db.session.add(patient)
            db.session.commit()
            print(f"✅ Created new user: {email}")
        else:
            print(f"✅ User exists: {email}")

        # Create JWT tokens
        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))

        # Redirect to frontend with tokens
        picture_url = picture or ""
        redirect_url = f"{FE_URL}/auth/callback?access_token={quote(access_token)}&refresh_token={quote(refresh_token)}&picture={quote(picture_url)}"
        
        print(f"🔗 Redirecting to: {redirect_url}")
        return redirect(redirect_url)

    except Exception as e:
        db.session.rollback()
        print(f"❌ Unexpected error in Google callback: {str(e)}")
        import traceback
        traceback.print_exc()
        error_url = f"{FE_URL}/login?error=server_error"
        return redirect(error_url)
