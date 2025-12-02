import React, { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { useToast } from '../../context/ToastContext';

function AuthCallback() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { setUser } = useAuth();
  const toast = useToast();

  useEffect(() => {
    const accessToken = searchParams.get('access_token');
    const refreshToken = searchParams.get('refresh_token');
    const picture = searchParams.get('picture');
    const error = searchParams.get('error');

    if (error) {
      console.error('OAuth Error:', error);
      toast.error('Google login failed. Please try again.');
      navigate('/login');
      return;
    }

    if (accessToken) {
      // Save tokens first
      localStorage.setItem('token', accessToken);
      if (refreshToken) {
        localStorage.setItem('refresh_token', refreshToken);
      }
      if (picture) {
        localStorage.setItem('picture', picture);
      }

      // Fetch user info with axios (will use token from localStorage)
      import('axios').then(({ default: axios }) => {
        // Set authorization header
        axios.defaults.headers.common['Authorization'] = `Bearer ${accessToken}`;
        
        axios.get(`${process.env.REACT_APP_API_URL || 'http://localhost:5001/api'}/auth/me`)
          .then(response => {
            const userData = response.data;
            localStorage.setItem('user', JSON.stringify(userData));
            setUser(userData);
            
            toast.success(`Welcome, ${userData.full_name}!`);
            
            // Redirect based on role with page reload to ensure AuthContext updates
            setTimeout(() => {
              if (userData.role === 'patient') {
                window.location.href = '/patient';
              } else if (userData.role === 'doctor') {
                window.location.href = '/doctor';
              } else if (userData.role === 'admin') {
                window.location.href = '/admin';
              } else {
                window.location.href = '/';
              }
            }, 500);
          })
          .catch(err => {
            console.error('Failed to get user info:', err);
            toast.error('Failed to get user information');
            navigate('/login');
          });
      });
    } else {
      toast.error('No authentication token received');
      navigate('/login');
    }
  }, [searchParams, navigate, setUser, toast]);

  return null; // No UI, just redirect
}

export default AuthCallback;
