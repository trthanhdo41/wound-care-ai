import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import './Login.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001/api';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  
  const { login } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const result = await login(email, password);
    
    if (result.success) {
      toast.success(`Welcome back, ${result.user.full_name}!`);
      redirectUser(result.user.role);
    } else {
      toast.error(result.error || 'Login failed. Please try again.');
    }
    
    setLoading(false);
  };

  const handleGoogleLogin = () => {
    setGoogleLoading(true);
    // Redirect to backend OAuth endpoint
    window.location.href = `${API_URL}/auth/login_by_google`;
  };

  const redirectUser = (role) => {
    setTimeout(() => {
      if (role === 'patient') {
        navigate('/patient');
      } else if (role === 'doctor') {
        navigate('/doctor');
      } else if (role === 'admin') {
        navigate('/admin');
      }
    }, 500);
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <div className="login-header">
          <h1>🩺 Wound Care AI</h1>
          <p>Diabetic Foot Ulcer Analysis System</p>
        </div>
        
        <form onSubmit={handleSubmit} className="login-form">
          <h2>Login</h2>
          
          <div className="form-group">
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              required
            />
          </div>
          
          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
            />
          </div>
          
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Logging in...' : 'Login'}
          </button>

          <div className="divider">
            <span>OR</span>
          </div>

          <div className="google-login-wrapper">
            <button 
              type="button"
              onClick={handleGoogleLogin}
              disabled={googleLoading}
              className="btn-google"
            >
              {googleLoading ? (
                'Redirecting to Google...'
              ) : (
                <>
                  <img
                    src="https://www.svgrepo.com/show/475656/google-color.svg"
                    alt="Google logo"
                    style={{ width: '20px', height: '20px', marginRight: '10px' }}
                  />
                  Continue with Google
                </>
              )}
            </button>
          </div>
          
          <div className="login-footer">
            <p>Don't have an account? <Link to="/register">Register here</Link></p>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Login;

