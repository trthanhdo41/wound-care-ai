import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import { FiArrowLeft, FiCalendar, FiFileText } from 'react-icons/fi';
import './PatientDetail.css';

function PatientDetail() {
  const { patientId } = useParams();
  const { API_URL } = useAuth();
  const navigate = useNavigate();
  const [analyses, setAnalyses] = useState([]);
  const [showRecommendationForm, setShowRecommendationForm] = useState(false);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [formData, setFormData] = useState({
    recommendation_text: '',
    treatment_plan: '',
    follow_up_date: ''
  });

  useEffect(() => {
    fetchAnalyses();
  }, [patientId]);

  const fetchAnalyses = async () => {
    try {
      const response = await axios.get(`${API_URL}/doctor/patient/${patientId}/analyses`);
      setAnalyses(response.data);
    } catch (error) {
      console.error('Error fetching analyses:', error);
    }
  };

  const handleAddRecommendation = (analysis) => {
    setSelectedAnalysis(analysis);
    setShowRecommendationForm(true);
  };

  const handleSubmitRecommendation = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${API_URL}/doctor/recommendations/add`, {
        analysis_id: selectedAnalysis.id,
        ...formData
      });
      alert('Recommendation added successfully');
      setShowRecommendationForm(false);
      setFormData({ recommendation_text: '', treatment_plan: '', follow_up_date: '' });
    } catch (error) {
      console.error('Error adding recommendation:', error);
      alert('Error adding recommendation');
    }
  };

  return (
    <div className="patient-detail-page">
      <div className="page-header">
        <button className="btn-back" onClick={() => navigate(-1)}>
          <FiArrowLeft /> Back
        </button>
        <h2>Patient Analysis History</h2>
      </div>

      <div className="analyses-timeline">
        {analyses.map((analysis) => (
          <div key={analysis.id} className="analysis-card">
            <div className="analysis-header">
              <div>
                <h3>Analysis #{analysis.id}</h3>
                <p>{new Date(analysis.created_at).toLocaleString()}</p>
              </div>
              <span className={`risk-badge ${analysis.risk_level}`}>
                {analysis.risk_level?.toUpperCase()}
              </span>
            </div>
            <div className="analysis-body">
              <p>Risk Score: {analysis.risk_score}/100</p>
              {analysis.image_path && (
                <img
                  src={`${API_URL}/analysis/image/${analysis.id}/segmented`}
                  alt="Analysis"
                  className="analysis-thumb"
                />
              )}
            </div>
            <button
              className="btn-add-recommendation"
              onClick={() => handleAddRecommendation(analysis)}
            >
              <FiFileText /> Add Recommendation
            </button>
          </div>
        ))}
      </div>

      {showRecommendationForm && (
        <div className="modal-overlay" onClick={() => setShowRecommendationForm(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Add Recommendation</h3>
            <form onSubmit={handleSubmitRecommendation}>
              <textarea
                placeholder="Recommendation"
                required
                value={formData.recommendation_text}
                onChange={(e) => setFormData({...formData, recommendation_text: e.target.value})}
              />
              <textarea
                placeholder="Treatment Plan"
                value={formData.treatment_plan}
                onChange={(e) => setFormData({...formData, treatment_plan: e.target.value})}
              />
              <input
                type="date"
                value={formData.follow_up_date}
                onChange={(e) => setFormData({...formData, follow_up_date: e.target.value})}
              />
              <div className="modal-actions">
                <button type="button" onClick={() => setShowRecommendationForm(false)}>Cancel</button>
                <button type="submit" className="btn-primary">Submit</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default PatientDetail;
