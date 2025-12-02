import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { FiUsers, FiCalendar, FiBarChart2, FiLogOut, FiPlus, FiEdit2, FiTrash2, FiLock, FiUnlock } from 'react-icons/fi';
import { FaHeartbeat } from 'react-icons/fa';
import './AdminDashboard.css';

function AdminDashboard() {
  const { user, logout, API_URL } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('users');
  const [users, setUsers] = useState([]);
  const [appointments, setAppointments] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [userRole, setUserRole] = useState('patient');
  const [showModal, setShowModal] = useState(false);
  const [modalType, setModalType] = useState(''); // 'user' or 'appointment'
  const [formData, setFormData] = useState({});

  useEffect(() => {
    if (!user || user.role !== 'admin') {
      navigate('/login');
    }
  }, [user, navigate]);

  useEffect(() => {
    if (activeTab === 'users') {
      fetchUsers();
    } else if (activeTab === 'appointments') {
      fetchAppointments();
    } else if (activeTab === 'statistics') {
      fetchStatistics();
    }
  }, [activeTab, userRole]);

  const fetchUsers = async () => {
    try {
      const response = await axios.get(`${API_URL}/admin/users?role=${userRole}`);
      setUsers(response.data);
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  const fetchAppointments = async () => {
    try {
      const response = await axios.get(`${API_URL}/admin/appointments`);
      setAppointments(response.data);
    } catch (error) {
      console.error('Error fetching appointments:', error);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await axios.get(`${API_URL}/admin/statistics`);
      setStatistics(response.data);
    } catch (error) {
      console.error('Error fetching statistics:', error);
    }
  };

  const handleCreateUser = () => {
    setModalType('user');
    setFormData({ role: userRole });
    setShowModal(true);
  };

  const handleCreateAppointment = () => {
    setModalType('appointment');
    setFormData({});
    setShowModal(true);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (modalType === 'user') {
        await axios.post(`${API_URL}/admin/users`, formData);
        fetchUsers();
      } else if (modalType === 'appointment') {
        await axios.post(`${API_URL}/admin/appointments`, formData);
        fetchAppointments();
      }
      setShowModal(false);
      setFormData({});
    } catch (error) {
      console.error('Error submitting:', error);
      alert(error.response?.data?.error || 'Error occurred');
    }
  };

  const handleDeleteUser = async (userId) => {
    if (!window.confirm('Are you sure you want to delete this user?')) return;
    try {
      await axios.delete(`${API_URL}/admin/users/${userId}`);
      fetchUsers();
    } catch (error) {
      console.error('Error deleting user:', error);
    }
  };

  const handleToggleLock = async (userId) => {
    try {
      await axios.post(`${API_URL}/admin/users/${userId}/lock`);
      fetchUsers();
    } catch (error) {
      console.error('Error toggling lock:', error);
    }
  };

  const handleDeleteAppointment = async (appointmentId) => {
    if (!window.confirm('Are you sure you want to delete this appointment?')) return;
    try {
      await axios.delete(`${API_URL}/admin/appointments/${appointmentId}`);
      fetchAppointments();
    } catch (error) {
      console.error('Error deleting appointment:', error);
    }
  };

  return (
    <div className="admin-dashboard">
      <div className="admin-sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon">
              <FaHeartbeat />
            </div>
            <h1>Admin Panel</h1>
          </div>
        </div>

        <div className="sidebar-nav">
          <button
            className={`nav-item ${activeTab === 'users' ? 'active' : ''}`}
            onClick={() => setActiveTab('users')}
          >
            <FiUsers />
            <span>Users</span>
          </button>
          <button
            className={`nav-item ${activeTab === 'appointments' ? 'active' : ''}`}
            onClick={() => setActiveTab('appointments')}
          >
            <FiCalendar />
            <span>Appointments</span>
          </button>
          <button
            className={`nav-item ${activeTab === 'statistics' ? 'active' : ''}`}
            onClick={() => setActiveTab('statistics')}
          >
            <FiBarChart2 />
            <span>Statistics</span>
          </button>
        </div>

        <div className="sidebar-footer">
          <button className="nav-item logout-btn" onClick={logout}>
            <FiLogOut />
            <span>Logout</span>
          </button>
        </div>
      </div>

      <div className="admin-main">
        <div className="admin-header">
          <h2>
            {activeTab === 'users' && 'User Management'}
            {activeTab === 'appointments' && 'Appointment Management'}
            {activeTab === 'statistics' && 'Statistics'}
          </h2>
          {activeTab === 'users' && (
            <button className="btn-primary" onClick={handleCreateUser}>
              <FiPlus /> Create User
            </button>
          )}
          {activeTab === 'appointments' && (
            <button className="btn-primary" onClick={handleCreateAppointment}>
              <FiPlus /> Create Appointment
            </button>
          )}
        </div>

        <div className="admin-content">
          {activeTab === 'users' && (
            <>
              <div className="tabs">
                <button
                  className={userRole === 'patient' ? 'active' : ''}
                  onClick={() => setUserRole('patient')}
                >
                  Patients
                </button>
                <button
                  className={userRole === 'doctor' ? 'active' : ''}
                  onClick={() => setUserRole('doctor')}
                >
                  Doctors
                </button>
              </div>

              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user) => (
                    <tr key={user.id}>
                      <td>{user.id}</td>
                      <td>{user.full_name}</td>
                      <td>{user.email}</td>
                      <td>
                        <span className={`status ${user.is_active ? 'active' : 'inactive'}`}>
                          {user.is_active ? 'Active' : 'Locked'}
                        </span>
                      </td>
                      <td>{new Date(user.created_at).toLocaleDateString()}</td>
                      <td>
                        <div className="action-buttons">
                          <button
                            className="btn-icon"
                            onClick={() => handleToggleLock(user.id)}
                            title={user.is_active ? 'Lock' : 'Unlock'}
                          >
                            {user.is_active ? <FiLock /> : <FiUnlock />}
                          </button>
                          <button
                            className="btn-icon danger"
                            onClick={() => handleDeleteUser(user.id)}
                            title="Delete"
                          >
                            <FiTrash2 />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {activeTab === 'appointments' && (
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Patient</th>
                  <th>Doctor</th>
                  <th>Date</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {appointments.map((apt) => (
                  <tr key={apt.id}>
                    <td>{apt.id}</td>
                    <td>{apt.patient.name}</td>
                    <td>{apt.doctor.name}</td>
                    <td>{new Date(apt.appointment_date).toLocaleString()}</td>
                    <td>
                      <span className={`status ${apt.status}`}>{apt.status}</span>
                    </td>
                    <td>
                      <button
                        className="btn-icon danger"
                        onClick={() => handleDeleteAppointment(apt.id)}
                      >
                        <FiTrash2 />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {activeTab === 'statistics' && statistics && (
            <div className="stats-grid">
              <div className="stat-card">
                <h3>Total Patients</h3>
                <p className="stat-number">{statistics.total_patients}</p>
              </div>
              <div className="stat-card">
                <h3>Total Doctors</h3>
                <p className="stat-number">{statistics.total_doctors}</p>
              </div>
              <div className="stat-card">
                <h3>Total Analyses</h3>
                <p className="stat-number">{statistics.total_analyses}</p>
              </div>
              <div className="stat-card">
                <h3>Total Appointments</h3>
                <p className="stat-number">{statistics.total_appointments}</p>
              </div>
              <div className="stat-card full-width">
                <h3>Risk Distribution</h3>
                <div className="risk-bars">
                  <div className="risk-bar">
                    <span>Low</span>
                    <div className="bar low" style={{width: `${(statistics.risk_distribution.low / statistics.total_analyses) * 100}%`}}></div>
                    <span>{statistics.risk_distribution.low}</span>
                  </div>
                  <div className="risk-bar">
                    <span>Medium</span>
                    <div className="bar medium" style={{width: `${(statistics.risk_distribution.medium / statistics.total_analyses) * 100}%`}}></div>
                    <span>{statistics.risk_distribution.medium}</span>
                  </div>
                  <div className="risk-bar">
                    <span>High</span>
                    <div className="bar high" style={{width: `${(statistics.risk_distribution.high / statistics.total_analyses) * 100}%`}}></div>
                    <span>{statistics.risk_distribution.high}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>{modalType === 'user' ? 'Create User' : 'Create Appointment'}</h3>
            <form onSubmit={handleSubmit}>
              {modalType === 'user' && (
                <>
                  <input
                    type="email"
                    placeholder="Email"
                    required
                    value={formData.email || ''}
                    onChange={(e) => setFormData({...formData, email: e.target.value})}
                  />
                  <input
                    type="password"
                    placeholder="Password"
                    required
                    value={formData.password || ''}
                    onChange={(e) => setFormData({...formData, password: e.target.value})}
                  />
                  <input
                    type="text"
                    placeholder="Full Name"
                    required
                    value={formData.full_name || ''}
                    onChange={(e) => setFormData({...formData, full_name: e.target.value})}
                  />
                </>
              )}
              {modalType === 'appointment' && (
                <>
                  <input
                    type="number"
                    placeholder="Patient ID"
                    required
                    value={formData.patient_id || ''}
                    onChange={(e) => setFormData({...formData, patient_id: e.target.value})}
                  />
                  <input
                    type="number"
                    placeholder="Doctor ID"
                    required
                    value={formData.doctor_id || ''}
                    onChange={(e) => setFormData({...formData, doctor_id: e.target.value})}
                  />
                  <input
                    type="datetime-local"
                    required
                    value={formData.appointment_date || ''}
                    onChange={(e) => setFormData({...formData, appointment_date: e.target.value})}
                  />
                  <textarea
                    placeholder="Reason"
                    value={formData.reason || ''}
                    onChange={(e) => setFormData({...formData, reason: e.target.value})}
                  />
                </>
              )}
              <div className="modal-actions">
                <button type="button" onClick={() => setShowModal(false)}>Cancel</button>
                <button type="submit" className="btn-primary">Create</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default AdminDashboard;
