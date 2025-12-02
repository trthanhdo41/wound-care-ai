import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { FiSearch, FiUser, FiEye } from 'react-icons/fi';
import './DoctorPatients.css';

function DoctorPatients() {
  const { API_URL } = useAuth();
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchPatients();
  }, [search]);

  const fetchPatients = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/doctor/patients?search=${search}`);
      setPatients(response.data);
    } catch (error) {
      console.error('Error fetching patients:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleViewPatient = (patientId) => {
    navigate(`/doctor/patient/${patientId}`);
  };

  return (
    <div className="doctor-patients-page">
      <div className="page-header">
        <h2>My Patients</h2>
        <div className="search-box">
          <FiSearch />
          <input
            type="text"
            placeholder="Search by name, email, or ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      <div className="patients-grid">
        {loading ? (
          <p>Loading...</p>
        ) : patients.length === 0 ? (
          <p>No patients found</p>
        ) : (
          patients.map((patient) => (
            <div key={patient.id} className="patient-card">
              <div className="patient-avatar">
                <FiUser />
              </div>
              <div className="patient-info">
                <h3>{patient.full_name}</h3>
                <p>{patient.email}</p>
                <div className="patient-meta">
                  <span>Age: {patient.age || 'N/A'}</span>
                  <span>Gender: {patient.gender || 'N/A'}</span>
                  <span>Diabetes: {patient.diabetes_type || 'N/A'}</span>
                </div>
              </div>
              <button
                className="btn-view"
                onClick={() => handleViewPatient(patient.id)}
              >
                <FiEye /> View Details
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default DoctorPatients;
