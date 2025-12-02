import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import { FiUsers, FiActivity, FiAlertCircle, FiCheckCircle, FiLogOut, FiSearch, FiEye, FiTrendingUp, FiUser, FiMessageCircle, FiEdit2 } from 'react-icons/fi';
import { FaUserMd } from 'react-icons/fa';
import { EditProfile } from '../components/EditProfile';
import { MedicalChatBot } from '../components/MedicalChatBot';
import { ChatWindow } from '../components/ChatWindow';
import { DoctorChart } from '../components/DoctorChart';
import './DoctorDashboard.css';

function DoctorDashboard() {
  const { user, logout, API_URL } = useAuth();
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientAnalyses, setPatientAnalyses] = useState([]);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showEditProfile, setShowEditProfile] = useState(false);
  const [allAnalyses, setAllAnalyses] = useState([]);

  useEffect(() => {
    fetchPatients();
    fetchAllAnalyses();
  }, []);

  const fetchPatients = async () => {
    try {
      const response = await axios.get(`${API_URL}/doctors/patients`);
      setPatients(response.data.patients);
    } catch (error) {
      console.error('Failed to fetch patients:', error);
    }
  };

  const fetchAllAnalyses = async () => {
    try {
      // Fetch all analyses from all patients
      const response = await axios.get(`${API_URL}/doctors/patients`);
      const patientsData = response.data.patients;
      
      let analyses = [];
      for (const patient of patientsData) {
        const analysisResponse = await axios.get(`${API_URL}/doctors/patients/${patient.id}/analyses`);
        analyses = [...analyses, ...analysisResponse.data.analyses];
      }
      
      setAllAnalyses(analyses);
    } catch (error) {
      console.error('Failed to fetch all analyses:', error);
    }
  };

  const viewPatientAnalyses = async (patientId) => {
    try {
      const response = await axios.get(`${API_URL}/doctors/patients/${patientId}/analyses`);
      setPatientAnalyses(response.data.analyses);
      setSelectedPatient(patientId);
      setActiveTab('analyses');
    } catch (error) {
      console.error('Failed to fetch analyses:', error);
    }
  };

  const getRiskColor = (level) => {
    const colors = {
      'Low': '#10b981',
      'Medium': '#f59e0b',
      'High': '#ef4444',
      'Critical': '#dc2626',
      'low': '#10b981',
      'medium': '#f59e0b',
      'high': '#ef4444',
      'critical': '#dc2626'
    };
    return colors[level] || '#6b7280';
  };

  const getRiskIcon = (level) => {
    if (level === 'Low' || level === 'low') return <FiCheckCircle />;
    if (level === 'Medium' || level === 'medium') return <FiAlertCircle />;
    return <FiAlertCircle />;
  };

  return (
    <div className="dashboard-wrapper">
      {/* Sidebar */}
      <aside className="dashboard-sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon">
              <FaUserMd />
            </div>
            <h1>WoundCare AI</h1>
          </div>
          <p className="logo-subtitle">Doctor Portal</p>
        </div>

        <nav className="sidebar-nav">
          <button 
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <FiActivity /> <span>Dashboard</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'patients' ? 'active' : ''}`}
            onClick={() => setActiveTab('patients')}
          >
            <FiUsers /> <span>Patients</span>
          </button>
          {selectedPatient && (
            <button 
              className={`nav-item ${activeTab === 'analyses' ? 'active' : ''}`}
              onClick={() => setActiveTab('analyses')}
            >
              <FiActivity /> <span>Patient Analyses</span>
            </button>
          )}
          <button 
            className={`nav-item ${activeTab === 'messages' ? 'active' : ''}`}
            onClick={() => setActiveTab('messages')}
          >
            <FiMessageCircle /> <span>Messages</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'profile' ? 'active' : ''}`}
            onClick={() => setActiveTab('profile')}
          >
            <FiUser /> <span>Profile</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <button onClick={logout} className="logout-btn">
            <FiLogOut /> <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="dashboard-main">
        <header className="dashboard-header">
          <div>
            <h2>Welcome back, Dr. {user?.full_name}</h2>
            <p className="header-subtitle">Manage and monitor your patients' wound healing progress</p>
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
            <div>
              <div className="content-header">
                <h3>Overview</h3>
              </div>

              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-icon" style={{ background: '#dbeafe' }}>
                    <FiUsers style={{ color: '#3b82f6' }} />
                  </div>
                  <div className="stat-info">
                    <p className="stat-label">Total Patients</p>
                    <h3 className="stat-value">{patients.length}</h3>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon" style={{ background: '#d1fae5' }}>
                    <FiActivity style={{ color: '#10b981' }} />
                  </div>
                  <div className="stat-info">
                    <p className="stat-label">Total Analyses</p>
                    <h3 className="stat-value">{allAnalyses.length}</h3>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon" style={{ background: '#fecaca' }}>
                    <FiAlertCircle style={{ color: '#ef4444' }} />
                  </div>
                  <div className="stat-info">
                    <p className="stat-label">High Risk Cases</p>
                    <h3 className="stat-value">
                      {allAnalyses.filter(a => a.risk_level === 'High' || a.risk_level === 'Critical' || a.risk_level === 'high' || a.risk_level === 'critical').length}
                    </h3>
                  </div>
                </div>
              </div>

              {/* Chart */}
              {allAnalyses.length > 0 && (
                <div className="chart-card">
                  <h3>Analysis Trends (Last 7 Days)</h3>
                  <p className="chart-subtitle">Track patient analyses and risk levels over time</p>
                  <DoctorChart analyses={allAnalyses} />
                </div>
              )}

              <div className="recent-patients">
                <h3>Recent Patients</h3>
                {patients.slice(0, 5).map((patient) => (
                  <div key={patient.id} className="recent-patient-item" onClick={() => {
                    setSelectedPatient(patient.id);
                    viewPatientAnalyses(patient.id);
                  }}>
                    <div className="recent-patient-avatar">
                      <FiUser />
                    </div>
                    <div className="recent-patient-info">
                      <p className="recent-patient-name">{patient.full_name || `Patient #${patient.id}`}</p>
                      <p className="recent-patient-meta">{patient.analysis_count || 0} analyses</p>
                    </div>
                    {patient.latest_risk_level && (
                      <div className="recent-patient-badge" style={{ background: getRiskColor(patient.latest_risk_level) }}>
                        {patient.latest_risk_level}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Patients Tab */}
          {activeTab === 'patients' && (
            <div>
              <div className="content-header">
                <h3>Patient Management</h3>
                <div className="search-box">
                  <FiSearch />
                  <input type="text" placeholder="Search patients..." />
                </div>
              </div>

              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-icon" style={{ background: '#dbeafe' }}>
                    <FiUsers style={{ color: '#3b82f6' }} />
                  </div>
                  <div className="stat-info">
                    <p className="stat-label">Total Patients</p>
                    <h3 className="stat-value">{patients.length}</h3>
                  </div>
                </div>
              </div>

              <div className="patients-grid">
                {patients.map((patient) => (
                  <div key={patient.id} className="patient-card-modern">
                    <div className="patient-header">
                      <div className="patient-avatar">
                        <FiUser />
                      </div>
                      <div className="patient-info">
                        <h4>{patient.full_name || `Patient #${patient.id}`}</h4>
                        <p className="patient-meta">{patient.email || 'No email'}</p>
                      </div>
                    </div>
                    
                    <div className="patient-stats">
                      <div className="patient-stat-item">
                        <FiActivity className="stat-item-icon" />
                        <div>
                          <p className="stat-item-label">Total Analyses</p>
                          <p className="stat-item-value">{patient.analysis_count || 0}</p>
                        </div>
                      </div>
                      
                      {patient.latest_risk_level && (
                        <div className="patient-stat-item">
                          {getRiskIcon(patient.latest_risk_level)}
                          <div>
                            <p className="stat-item-label">Latest Risk</p>
                            <p className="stat-item-value" style={{ color: getRiskColor(patient.latest_risk_level) }}>
                              {patient.latest_risk_level}
                            </p>
                          </div>
                        </div>
                      )}
                    </div>

                    {patient.last_analysis_date && (
                      <p className="patient-last-visit">
                        Last analysis: {new Date(patient.last_analysis_date).toLocaleDateString()}
                      </p>
                    )}
                    
                    <button 
                      onClick={() => viewPatientAnalyses(patient.id)}
                      className="btn-view-analyses"
                    >
                      <FiEye /> View Analyses ({patient.analysis_count || 0})
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Analyses Tab */}
          {activeTab === 'analyses' && selectedPatient && (
            <div>
              <div className="content-header">
                <h3>Patient #{selectedPatient} - Analysis History</h3>
                <button 
                  onClick={() => setActiveTab('patients')}
                  className="btn-back"
                >
                  ← Back to Patients
                </button>
              </div>

              {patientAnalyses.length === 0 ? (
                <div className="empty-state">
                  <FiActivity className="empty-icon" />
                  <p>No analyses found for this patient</p>
                </div>
              ) : (
                <div className="analyses-grid">
                  {patientAnalyses.map((analysis) => (
                    <div key={analysis.id} className="analysis-card-modern">
                      <div className="analysis-header">
                        <div className="analysis-date">
                          <FiActivity />
                          <span>{new Date(analysis.created_at).toLocaleDateString()}</span>
                        </div>
                        <div 
                          className="risk-badge" 
                          style={{ background: getRiskColor(analysis.risk_level) }}
                        >
                          {getRiskIcon(analysis.risk_level)}
                          {analysis.risk_level}
                        </div>
                      </div>

                      <div className="analysis-metrics">
                        <div className="metric-item">
                          <div className="metric-icon" style={{ background: '#dbeafe' }}>
                            <FiActivity style={{ color: '#3b82f6' }} />
                          </div>
                          <div className="metric-info">
                            <p className="metric-label">Wound Area</p>
                            <p className="metric-value">{analysis.wound_area_cm2 || 'N/A'} cm²</p>
                          </div>
                        </div>

                        <div className="metric-item">
                          <div className="metric-icon" style={{ background: '#fef3c7' }}>
                            <FiTrendingUp style={{ color: '#f59e0b' }} />
                          </div>
                          <div className="metric-info">
                            <p className="metric-label">Risk Score</p>
                            <p className="metric-value">{analysis.risk_score || 'N/A'}/100</p>
                          </div>
                        </div>
                      </div>

                      {analysis.color_analysis && (
                        <div className="analysis-details">
                          <p className="detail-title">Color Analysis:</p>
                          <p className="detail-text">
                            {typeof analysis.color_analysis === 'string' 
                              ? JSON.parse(analysis.color_analysis).tissue_types 
                                ? Object.values(JSON.parse(analysis.color_analysis).tissue_types)[0]
                                : 'Available'
                              : 'Available'}
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Messages Tab */}
          {activeTab === 'messages' && (
            <div>
              <div className="content-header">
                <h3>Patient Messages</h3>
              </div>
              <div className="chat-container-full">
                <ChatWindow />
              </div>
            </div>
          )}

          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div>
              <div className="content-header">
                <h3>Doctor Profile</h3>
              </div>

              <div className="profile-view">
                <div className="profile-card">
                  <div className="profile-header">
                    <div className="profile-avatar-large">
                      <FiUser />
                    </div>
                    <h3>{user?.full_name}</h3>
                    <p className="profile-role">Doctor</p>
                    <button 
                      onClick={() => setShowEditProfile(true)}
                      className="btn-edit-profile"
                    >
                      <FiEdit2 /> Edit Profile
                    </button>
                  </div>
                  
                  <div className="profile-details">
                    <div className="profile-detail-item">
                      <span className="detail-label">Email:</span>
                      <span className="detail-value">{user?.email}</span>
                    </div>
                    {user?.phone && (
                      <div className="profile-detail-item">
                        <span className="detail-label">Phone:</span>
                        <span className="detail-value">{user?.phone}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Chatbot */}
      <MedicalChatBot />

      {/* Edit Profile Modal */}
      {showEditProfile && (
        <EditProfile 
          user={user}
          onSave={async (formData) => {
            try {
              await axios.put(`${API_URL}/auth/profile`, formData);
              setShowEditProfile(false);
              window.location.reload();
            } catch (error) {
              console.error('Failed to update profile:', error);
              alert('Failed to update profile');
            }
          }}
          onCancel={() => setShowEditProfile(false)}
        />
      )}
    </div>
  );
}

export default DoctorDashboard;

