import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import axios from 'axios';
import { 
  FiUpload, FiActivity, FiUser, FiLogOut, 
  FiImage, FiCheckCircle, FiAlertTriangle, FiAlertCircle, FiClock, FiTrendingUp,
  FiX, FiEye, FiEdit2, FiDownload, FiMessageCircle
} from 'react-icons/fi';
import { MdDashboard } from 'react-icons/md';
import { FaHeartbeat } from 'react-icons/fa';
import { WoundChart } from '../components/WoundChart';
import { EditProfile } from '../components/EditProfile';
import { MedicalChatBot } from '../components/MedicalChatBot';
import { ChatWindow } from '../components/ChatWindow';
import { TissueCompositionChart, TextureAnalysisChart } from '../components/AnalysisCharts';
import './PatientDashboard.css';

function PatientDashboard() {
  const { user, logout, API_URL } = useAuth();
  const toast = useToast();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [analyses, setAnalyses] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [showEditProfile, setShowEditProfile] = useState(false);

  useEffect(() => {
    fetchAnalysisHistory();
  }, []);

  const fetchAnalysisHistory = async () => {
    try {
      const response = await axios.get(`${API_URL}/analysis/history`);
      setAnalyses(response.data.analyses || []);
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File size must be less than 10MB');
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.warning('Please select an image first');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/analysis/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setAnalysisResult(response.data);
      toast.success('Analysis completed successfully!');
      fetchAnalysisHistory();
      setActiveTab('results');
    } catch (error) {
      toast.error(error.response?.data?.error || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };



  const getRiskColor = (level) => {
    const colors = {
      low: '#10b981',
      medium: '#f59e0b',
      high: '#ef4444',
      critical: '#dc2626'
    };
    return colors[level] || '#6b7280';
  };

  const getRiskIcon = (level) => {
    if (level === 'low') return <FiCheckCircle />;
    if (level === 'critical' || level === 'high') return <FiAlertTriangle />;
    return <FiClock />;
  };

  const clearUpload = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
  };

  const handleSaveProfile = async (formData) => {
    try {
      await axios.put(`${API_URL}/auth/profile`, formData);
      toast.success('Profile updated successfully!');
      setShowEditProfile(false);
      // Update user context if needed
      window.location.reload();
    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to update profile');
      throw error;
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      
      // Check if it's an image
      if (!file.type.startsWith('image/')) {
        toast.error('Please upload an image file');
        return;
      }
      
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File size must be less than 10MB');
        return;
      }
      
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisResult(null);
    }
  };

  return (
    <div className="patient-dashboard">
      {/* Sidebar */}
      <aside className="dashboard-sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon">
              <FaHeartbeat />
            </div>
            <h1>WoundCare AI</h1>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <MdDashboard /> <span>Dashboard</span>
          </button>
          <button
            className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <FiUpload /> <span>New Analysis</span>
          </button>
          <button
            className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <FiActivity /> <span>History</span>
          </button>
          <button
            className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <FiMessageCircle /> <span>Chat with Doctor</span>
          </button>
          <button
            className={`nav-item ${activeTab === 'profile' ? 'active' : ''}`}
            onClick={() => setActiveTab('profile')}
          >
            <FiUser /> <span>Profile</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <button onClick={logout} className="btn-logout">
            <FiLogOut /> <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="dashboard-main">
        <header className="dashboard-header">
          <div>
            <h2>Welcome back, {user?.full_name}</h2>
            <p className="header-subtitle">Monitor and analyze your wound healing progress</p>
          </div>
          <div className="header-user">
            <div className="user-avatar">
              <FiUser />
            </div>
          </div>
        </header>

        <div className="dashboard-content">
          {/* Dashboard Tab */}
          {activeTab === 'dashboard' && (
            <div className="dashboard-grid">
              <div className="stat-card">
                <div className="stat-icon" style={{ background: '#e0f2fe' }}>
                  <FiActivity style={{ color: '#0ea5e9' }} />
                </div>
                <div className="stat-info">
                  <p className="stat-label">Total Analyses</p>
                  <h3 className="stat-value">{analyses.length}</h3>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon" style={{ background: '#d1fae5' }}>
                  <FiCheckCircle style={{ color: '#10b981' }} />
                </div>
                <div className="stat-info">
                  <p className="stat-label">Low Risk</p>
                  <h3 className="stat-value">
                    {analyses.filter(a => a.risk_level === 'low').length}
                  </h3>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon" style={{ background: '#fef3c7' }}>
                  <FiAlertTriangle style={{ color: '#f59e0b' }} />
                </div>
                <div className="stat-info">
                  <p className="stat-label">Medium Risk</p>
                  <h3 className="stat-value">
                    {analyses.filter(a => a.risk_level === 'medium').length}
                  </h3>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-icon" style={{ background: '#fecaca' }}>
                  <FiTrendingUp style={{ color: '#ef4444' }} />
                </div>
                <div className="stat-info">
                  <p className="stat-label">High Risk</p>
                  <h3 className="stat-value">
                    {analyses.filter(a => a.risk_level === 'high' || a.risk_level === 'critical').length}
                  </h3>
                </div>
              </div>

              {/* Healing Progress Chart */}
              <div className="chart-card">
                <h3>Healing Progress</h3>
                <p className="chart-subtitle">Track your wound healing over time</p>
                {analyses.length > 0 ? (
                  <WoundChart analyses={analyses} />
                ) : (
                  <div className="chart-empty-state">
                    <FiTrendingUp className="chart-empty-icon" />
                    <p>No data yet. Upload your first wound image to start tracking progress.</p>
                    <button onClick={() => setActiveTab('upload')} className="btn-chart-upload">
                      <FiUpload /> Upload Image
                    </button>
                  </div>
                )}
              </div>

              <div className="recent-analyses">
                <h3>Recent Analyses</h3>
                {analyses.length > 0 ? (
                  analyses.slice(0, 5).map((analysis) => (
                    <div key={analysis.id} className="recent-item">
                      <div className="recent-icon" style={{ color: getRiskColor(analysis.risk_level) }}>
                        {getRiskIcon(analysis.risk_level)}
                      </div>
                      <div className="recent-info">
                        <p className="recent-title">Analysis #{analysis.id}</p>
                        <p className="recent-date">{new Date(analysis.created_at).toLocaleDateString()}</p>
                      </div>
                      <div className="recent-badge" style={{ background: getRiskColor(analysis.risk_level) }}>
                        {analysis.risk_level}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="recent-empty">
                    <FiActivity className="recent-empty-icon" />
                    <p>No analysis history yet</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div className="upload-container">
              <div className="upload-card">
                <h3>Upload Wound Image</h3>
                <p className="upload-subtitle">Upload a clear image of the wound for AI analysis</p>

                <div className="upload-area">
                  {!previewUrl ? (
                    <label 
                      htmlFor="file-input" 
                      className="upload-dropzone"
                      onDragOver={handleDragOver}
                      onDragEnter={handleDragEnter}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                    >
                      <FiImage className="upload-icon" />
                      <p className="upload-text">Click to upload or drag and drop</p>
                      <p className="upload-hint">PNG, JPG up to 10MB</p>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileSelect}
                        id="file-input"
                        hidden
                      />
                    </label>
                  ) : (
                    <div className="preview-container">
                      <button className="btn-clear" onClick={clearUpload}>
                        <FiX />
                      </button>
                      <img src={previewUrl} alt="Preview" className="preview-image" />
                      <p className="preview-name">{selectedFile?.name}</p>
                    </div>
                  )}
                </div>

                <button
                  onClick={handleUpload}
                  disabled={!selectedFile || uploading}
                  className="btn-analyze"
                >
                  {uploading ? (
                    <>
                      <div className="spinner"></div> Analyzing...
                    </>
                  ) : (
                    <>
                      <FiUpload /> Analyze Image
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && analysisResult && (
            <div className="results-container">
              <h3>Analysis Results</h3>
              
              {/* Image Comparison - 4 images */}
              {analysisResult.images && previewUrl && (
                <div className="image-comparison-four">
                  <div className="comparison-item">
                    <h4>1) Original Image</h4>
                    <img 
                      src={previewUrl} 
                      alt="Original" 
                      className="comparison-image"
                    />
                  </div>
                  <div className="comparison-item">
                    <h4>2) AI Segmentation Mask</h4>
                    <img 
                      src={`http://localhost:5001${analysisResult.images.segmented}`} 
                      alt="Segmented" 
                      className="comparison-image"
                    />
                  </div>
                  <div className="comparison-item">
                    <h4>3) Wound Zoom</h4>
                    <img 
                      src={`http://localhost:5001${analysisResult.images.wound_zoom}`} 
                      alt="Wound Zoom" 
                      className="comparison-image"
                    />
                  </div>
                  <div className="comparison-item">
                    <h4>4) AI Attention (High Focus)</h4>
                    <img 
                      src={`http://localhost:5001${analysisResult.images.gradcam}`} 
                      alt="Grad-CAM Attention Map" 
                      className="comparison-image"
                    />
                  </div>
                </div>
              )}

              {/* Risk Level Bar */}
              <div className="risk-level-section">
                <h4>7) Risk Level</h4>
                <div className="risk-level-bar">
                  <div className="risk-bar-container">
                    <div className="risk-bar-gradient"></div>
                    <div 
                      className="risk-indicator-arrow" 
                      style={{ left: `${analysisResult.results.risk_assessment.risk_score}%` }}
                    >
                      <div className="arrow-pointer"></div>
                      <div className="arrow-label">{analysisResult.results.risk_assessment.risk_score}</div>
                    </div>
                  </div>
                  <div className="risk-bar-labels">
                    <span>LOW</span>
                    <span>MEDIUM</span>
                    <span>HIGH</span>
                  </div>
                </div>
              </div>

              {/* Risk Assessment Card - Small card below Risk Level Bar */}
              <div className="results-grid-single">
                <div className="result-card-modern">
                  <div className="result-header">
                    <div className="result-icon" style={{ background: getRiskColor(analysisResult.results.risk_assessment.risk_level) + '20' }}>
                      {getRiskIcon(analysisResult.results.risk_assessment.risk_level)}
                    </div>
                    <h4>Risk Assessment</h4>
                  </div>
                  <div className="result-value" style={{ color: getRiskColor(analysisResult.results.risk_assessment.risk_level) }}>
                    {analysisResult.results.risk_assessment.risk_level.toUpperCase()}
                  </div>
                  <p className="result-detail">Score: {analysisResult.results.risk_assessment.risk_score}/100</p>
                </div>
              </div>

              {/* Charts Section */}
              <div className="charts-section">
                <div className="chart-container">
                  <TissueCompositionChart colorAnalysis={analysisResult.results.color_analysis} />
                </div>
                <div className="chart-container">
                  <TextureAnalysisChart roughnessAnalysis={analysisResult.results.roughness_analysis} />
                </div>
              </div>

              {/* Risk Assessment Card */}
              {analysisResult.results.risk_assessment && (
                <div className="risk-assessment-card-inline">
                  <div className="section-header">
                    <FaHeartbeat className="section-icon" />
                    <h3>7) Risk Assessment</h3>
                  </div>
                  <div className={`risk-level-badge ${analysisResult.results.risk_assessment.risk_level.toLowerCase()}`}>
                    {analysisResult.results.risk_assessment.risk_level === 'low' && <FiCheckCircle />}
                    {analysisResult.results.risk_assessment.risk_level === 'medium' && <FiAlertTriangle />}
                    {analysisResult.results.risk_assessment.risk_level === 'high' && <FiAlertCircle />}
                    <span className="risk-label">
                      {analysisResult.results.risk_assessment.risk_level.toUpperCase()} RISK
                    </span>
                    <span className="risk-score">
                      Score: {analysisResult.results.risk_assessment.risk_score}/100
                    </span>
                  </div>
                  <p className="risk-recommendation">
                    {analysisResult.results.risk_assessment.recommendation}
                  </p>
                </div>
              )}

              {/* Export PDF Button */}
              <div className="export-section">
                <button 
                  className="btn-export-pdf"
                  onClick={async () => {
                    try {
                      const response = await axios.get(
                        `${API_URL}/analysis/export-pdf/${analysisResult.analysis_id}`,
                        { responseType: 'blob' }
                      );
                      const url = window.URL.createObjectURL(new Blob([response.data]));
                      const link = document.createElement('a');
                      link.href = url;
                      link.setAttribute('download', `analysis_report_${analysisResult.analysis_id}.pdf`);
                      document.body.appendChild(link);
                      link.click();
                      link.remove();
                      toast.success('PDF downloaded successfully!');
                    } catch (error) {
                      toast.error('Failed to download PDF');
                    }
                  }}
                >
                  <FiDownload /> Download PDF Report
                </button>
              </div>

              {/* Action Plan Section - No AI Risk Assessment card here */}
              {analysisResult.results.risk_assessment && (
                <div className="action-plan-section">
                  <div className="section-header">
                    <FiCheckCircle className="section-icon" />
                    <h3>8) Action Plan & AI Recommendations</h3>
                  </div>
                  
                  <div className="action-plan-grid">

                    {/* AI Care Tips Card */}
                    <div className="action-card care-tips-card">
                      <div className="card-header">
                        <div className="header-icon success">
                          <FiCheckCircle />
                        </div>
                        <h4>AI-Generated Care Guidelines</h4>
                      </div>
                      <ul className="tips-list">
                        {analysisResult.results.risk_assessment?.care_guidelines ? (
                          analysisResult.results.risk_assessment.care_guidelines.map((guideline, index) => {
                            const iconMap = {
                              'check': FiCheckCircle,
                              'clock': FiClock,
                              'eye': FiEye,
                              'activity': FiActivity,
                              'alert': FiAlertCircle
                            };
                            const IconComponent = iconMap[guideline.icon] || FiCheckCircle;
                            
                            return (
                              <li key={index}>
                                <IconComponent className="tip-icon" />
                                <span>{guideline.text}</span>
                              </li>
                            );
                          })
                        ) : (
                          <>
                            <li>
                              <FiCheckCircle className="tip-icon" />
                              <span>Keep the wound clean and dry</span>
                            </li>
                            <li>
                              <FiClock className="tip-icon" />
                              <span>Follow prescribed medication schedule</span>
                            </li>
                            <li>
                              <FiEye className="tip-icon" />
                              <span>Monitor for signs of infection</span>
                            </li>
                            <li>
                              <FiActivity className="tip-icon" />
                              <span>Maintain healthy blood sugar levels</span>
                            </li>
                            <li>
                              <FiCheckCircle className="tip-icon" />
                              <span>Schedule regular follow-up appointments</span>
                            </li>
                          </>
                        )}
                      </ul>
                      <div className="ai-note">
                        <p>💡 <strong>Need more help?</strong> Click the chatbot icon below to ask AI specific questions about your wound care.</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <div className="history-container">
              <h3>Analysis History</h3>
              {analyses.length === 0 ? (
                <div className="empty-state">
                  <FiActivity className="empty-icon" />
                  <p>No analysis history yet</p>
                  <button onClick={() => setActiveTab('upload')} className="btn-primary-small">
                    Start First Analysis
                  </button>
                </div>
              ) : (
                <div className="history-grid">
                  {analyses.map((analysis) => (
                    <div key={analysis.id} className="history-card">
                      <div className="history-header">
                        <h4>Analysis #{analysis.id}</h4>
                        <div className="history-badge" style={{ background: getRiskColor(analysis.risk_level) }}>
                          {analysis.risk_level}
                        </div>
                      </div>
                      <div className="history-details">
                        <p><strong>Date:</strong> {new Date(analysis.created_at).toLocaleString()}</p>
                        <p><strong>Size:</strong> {analysis.wound_area_cm2} cm²</p>
                        <p><strong>Risk Score:</strong> {analysis.risk_score}/100</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="chat-container">
              <h3>Chat with Your Doctor</h3>
              <p className="chat-subtitle">Send messages and images to your doctor for consultation</p>
              <ChatWindow />
            </div>
          )}

          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="profile-container">
              <div className="profile-card">
                <div className="profile-header">
                  <div className="profile-avatar-large">
                    <FiUser />
                  </div>
                  <h3>{user?.full_name}</h3>
                  <p className="profile-role">{user?.role}</p>
                  <button onClick={() => setShowEditProfile(true)} className="btn-edit-profile">
                    <FiEdit2 /> Edit Profile
                  </button>
                </div>
                <div className="profile-details">
                  <div className="profile-item">
                    <label>Email</label>
                    <p>{user?.email}</p>
                  </div>
                  <div className="profile-item">
                    <label>Phone</label>
                    <p>{user?.phone || 'Not provided'}</p>
                  </div>
                  <div className="profile-item">
                    <label>Date of Birth</label>
                    <p>{user?.date_of_birth ? new Date(user.date_of_birth).toLocaleDateString() : 'Not provided'}</p>
                  </div>
                  <div className="profile-item">
                    <label>Address</label>
                    <p>{user?.address || 'Not provided'}</p>
                  </div>
                  <div className="profile-item">
                    <label>Member Since</label>
                    <p>{new Date(user?.created_at).toLocaleDateString()}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Edit Profile Modal */}
      {showEditProfile && (
        <EditProfile
          user={user}
          onSave={handleSaveProfile}
          onCancel={() => setShowEditProfile(false)}
        />
      )}

      {/* Medical ChatBot - Floating Button (like WikiTech) */}
      <MedicalChatBot />
    </div>
  );
}

export default PatientDashboard;
