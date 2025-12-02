import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  FiActivity, FiEye, FiTrendingUp, FiShield, 
  FiClock, FiCheckCircle, FiArrowRight, FiZap
} from 'react-icons/fi';
import { FaHeartbeat, FaRobot } from 'react-icons/fa';
import './LandingPage.css';

function LandingPage() {
  const navigate = useNavigate();

  const features = [
    {
      icon: <FiActivity />,
      title: 'AI-Powered Analysis',
      description: 'Advanced deep learning models analyze diabetic foot ulcer images with medical-grade accuracy'
    },
    {
      icon: <FiEye />,
      title: 'Real-time Monitoring',
      description: 'Track diabetic foot ulcer healing progress with detailed color analysis'
    },
    {
      icon: <FiTrendingUp />,
      title: 'Risk Assessment',
      description: 'Machine learning algorithms predict complications and healing outcomes'
    },
    {
      icon: <FaRobot />,
      title: 'AI Care Guidelines',
      description: 'Personalized treatment recommendations based on ulcer characteristics'
    },
    {
      icon: <FiClock />,
      title: 'Quick Results',
      description: 'Get comprehensive diabetic foot ulcer analysis in seconds, not hours'
    },
    {
      icon: <FiShield />,
      title: 'Secure & Private',
      description: 'Your medical data is encrypted and protected with industry standards'
    }
  ];

  const howItWorks = [
    {
      step: '1',
      title: 'Upload Image',
      description: 'Take a photo of the diabetic foot ulcer and upload it to our secure platform'
    },
    {
      step: '2',
      title: 'AI Analysis',
      description: 'Our AI models analyze color composition, size, and risk factors'
    },
    {
      step: '3',
      title: 'Get Results',
      description: 'Receive detailed reports with care guidelines and recommendations'
    },
    {
      step: '4',
      title: 'Track Progress',
      description: 'Monitor healing over time with visual charts and insights'
    }
  ];

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-badge">
            <FiZap className="badge-icon" />
            <span>AI-Powered Diabetic Foot Ulcer Care</span>
          </div>
          <h1 className="hero-title">
            Smart Diabetic Foot Ulcer Analysis
            <span className="gradient-text"> Powered by AI</span>
          </h1>
          <p className="hero-description">
            Advanced artificial intelligence technology for accurate diabetic foot ulcer assessment, 
            real-time monitoring, and personalized care recommendations.
          </p>
          <div className="hero-buttons">
            <button 
              className="btn-primary"
              onClick={() => navigate('/register')}
            >
              Get Started Free
              <FiArrowRight />
            </button>
            <button 
              className="btn-secondary"
              onClick={() => navigate('/login')}
            >
              Sign In
            </button>
          </div>
          <div className="hero-stats">
            <div className="stat-item">
              <FiCheckCircle className="stat-icon" />
              <div>
                <div className="stat-number">80%</div>
                <div className="stat-label">Accuracy Rate</div>
              </div>
            </div>
            <div className="stat-item">
              <FiClock className="stat-icon" />
              <div>
                <div className="stat-number">&lt;5s</div>
                <div className="stat-label">Analysis Time</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="section-header">
          <h2>Why Choose WoundCare AI?</h2>
          <p>Cutting-edge technology meets compassionate care</p>
        </div>
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <div className="feature-icon">{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works Section */}
      <section className="how-it-works-section">
        <div className="section-header">
          <h2>How It Works</h2>
          <p>Simple, fast, and accurate wound analysis in 4 easy steps</p>
        </div>
        <div className="steps-container">
          {howItWorks.map((item, index) => (
            <div key={index} className="step-card">
              <div className="step-number">{item.step}</div>
              <h3>{item.title}</h3>
              <p>{item.description}</p>
              {index < howItWorks.length - 1 && (
                <FiArrowRight className="step-arrow" />
              )}
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2>Ready to Transform Diabetic Foot Ulcer Care?</h2>
          <p>Join healthcare professionals using AI to improve patient outcomes</p>
          <button 
            className="btn-cta"
            onClick={() => navigate('/register')}
          >
            Start Your Free Trial
            <FiArrowRight />
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <FaHeartbeat className="footer-logo" />
            <span>WoundCare AI</span>
          </div>
          <p>© 2025 WoundCare AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default LandingPage;
