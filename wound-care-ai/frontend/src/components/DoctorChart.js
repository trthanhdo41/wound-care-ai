import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export const DoctorChart = ({ analyses }) => {
  // Group analyses by date
  const groupByDate = () => {
    const grouped = {};
    
    analyses.forEach(analysis => {
      const date = new Date(analysis.created_at).toLocaleDateString();
      if (!grouped[date]) {
        grouped[date] = {
          total: 0,
          high_risk: 0,
          medium_risk: 0,
          low_risk: 0
        };
      }
      
      grouped[date].total += 1;
      
      const risk = analysis.risk_level?.toLowerCase();
      if (risk === 'high' || risk === 'critical') {
        grouped[date].high_risk += 1;
      } else if (risk === 'medium') {
        grouped[date].medium_risk += 1;
      } else {
        grouped[date].low_risk += 1;
      }
    });
    
    return grouped;
  };

  const groupedData = groupByDate();
  const dates = Object.keys(groupedData).sort((a, b) => new Date(a) - new Date(b)).slice(-7);

  const data = {
    labels: dates,
    datasets: [
      {
        label: 'Total Analyses',
        data: dates.map(date => groupedData[date].total),
        borderColor: '#0ea5e9',
        backgroundColor: 'rgba(14, 165, 233, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'High Risk',
        data: dates.map(date => groupedData[date].high_risk),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Medium Risk',
        data: dates.map(date => groupedData[date].medium_risk),
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Low Risk',
        data: dates.map(date => groupedData[date].low_risk),
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.4,
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 15,
          font: {
            size: 12,
            family: "'Inter', sans-serif"
          }
        }
      },
      title: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        titleFont: {
          size: 13,
          family: "'Inter', sans-serif"
        },
        bodyFont: {
          size: 12,
          family: "'Inter', sans-serif"
        },
        cornerRadius: 8
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
          font: {
            size: 11,
            family: "'Inter', sans-serif"
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      x: {
        ticks: {
          font: {
            size: 11,
            family: "'Inter', sans-serif"
          }
        },
        grid: {
          display: false
        }
      }
    }
  };

  return (
    <div style={{ height: '300px', width: '100%' }}>
      <Line data={data} options={options} />
    </div>
  );
};
