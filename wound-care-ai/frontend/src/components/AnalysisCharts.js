import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartDataLabels
);

export const TissueCompositionChart = ({ colorAnalysis }) => {
  if (!colorAnalysis) {
    return null;
  }

  // Use pixel-based percentages if available (more accurate)
  let aggregatedData, aggregatedColors;
  
  if (colorAnalysis.pixel_based_percentages) {
    // Use pixel-based classification (accurate)
    aggregatedData = {
      'Red': colorAnalysis.pixel_based_percentages.Red || 0,
      'Yellow': colorAnalysis.pixel_based_percentages.Yellow || 0,
      'Dark': colorAnalysis.pixel_based_percentages.Dark || 0
    };
  } else {
    // Fallback to cluster-based aggregation
    const tissueTypes = colorAnalysis.tissue_types || {};
    const percentages = colorAnalysis.cluster_percentages || {};
    
    aggregatedData = {
      'Red': 0,
      'Yellow': 0,
      'Dark': 0
    };
    
    Object.keys(tissueTypes).forEach(key => {
      const tissueName = tissueTypes[key];
      const pctKey = `${key}_pct`;
      const pct = percentages[pctKey] || 0;
      
      if (aggregatedData[tissueName] !== undefined) {
        aggregatedData[tissueName] += pct;
      }
    });
  }
  
  // Fixed colors for tissue types
  aggregatedColors = {
    'Red': 'rgba(220, 38, 38, 0.8)',      // Red
    'Yellow': 'rgba(234, 179, 8, 0.8)',   // Yellow
    'Dark': 'rgba(55, 48, 163, 0.8)'      // Dark blue/purple
  };

  const labels = Object.keys(aggregatedData);
  const data = Object.values(aggregatedData);
  const colors = labels.map(label => aggregatedColors[label]);

  const chartData = {
    labels: labels,
    datasets: [{
      label: 'Color Analysis (%)',
      data: data,
      backgroundColor: colors,
      borderColor: colors.map(c => c.replace('0.8', '1')),
      borderWidth: 1,
      borderRadius: 0,
    }]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: '5) Color Analysis (%)',
        font: {
          size: 18,
          weight: 'bold',
          family: "'Inter', sans-serif"
        },
        padding: { bottom: 20 },
        align: 'center'
      },
      datalabels: {
        anchor: 'end',
        align: 'top',
        formatter: (value) => `${value.toFixed(1)}%`,
        font: {
          size: 13,
          weight: 'bold',
          family: "'Inter', sans-serif"
        },
        color: '#000'
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        cornerRadius: 8,
        callbacks: {
          label: function(context) {
            return `${context.parsed.y.toFixed(1)}%`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%';
          },
          font: {
            size: 11,
            family: "'Inter', sans-serif"
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      x: {
        ticks: {
          font: {
            size: 12,
            family: "'Inter', sans-serif",
            weight: '500'
          }
        },
        grid: {
          display: false
        }
      }
    }
  };

  return (
    <div style={{ height: '350px', width: '100%' }}>
      <Bar data={chartData} options={options} />
    </div>
  );
};

export const TextureAnalysisChart = ({ roughnessAnalysis }) => {
  if (!roughnessAnalysis) {
    return null;
  }

  const labels = ['Contrast', 'Homogeneity'];
  const data = [
    roughnessAnalysis.contrast || 0,
    roughnessAnalysis.homogeneity ? roughnessAnalysis.homogeneity * 1000 : 0 // Scale up for visibility
  ];

  const chartData = {
    labels: labels,
    datasets: [{
      label: 'Texture Metrics',
      data: data,
      backgroundColor: [
        'rgba(14, 165, 233, 0.8)',
        'rgba(16, 185, 129, 0.8)'
      ],
      borderColor: [
        'rgba(14, 165, 233, 1)',
        'rgba(16, 185, 129, 1)'
      ],
      borderWidth: 2,
      borderRadius: 8,
    }]
  };

  const options = {
    indexAxis: 'y', // Horizontal bar chart
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: '6) Texture Analysis',
        font: {
          size: 16,
          weight: 'bold',
          family: "'Inter', sans-serif"
        },
        padding: { bottom: 20 },
        align: 'start'
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        cornerRadius: 8,
        callbacks: {
          label: function(context) {
            if (context.label === 'Homogeneity') {
              return `Homogeneity: ${(context.parsed.x / 1000).toFixed(3)}`;
            }
            return `Contrast: ${context.parsed.x.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        ticks: {
          font: {
            size: 11,
            family: "'Inter', sans-serif"
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      y: {
        ticks: {
          font: {
            size: 12,
            family: "'Inter', sans-serif",
            weight: '500'
          }
        },
        grid: {
          display: false
        }
      }
    }
  };

  return (
    <div style={{ height: '200px', width: '100%' }}>
      <Bar data={chartData} options={options} />
    </div>
  );
};
