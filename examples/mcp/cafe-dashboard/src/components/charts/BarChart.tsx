"use client";

import { ChartData } from '@/lib/metrics';
import { Card } from '../ui/Card';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend,
  ChartOptions
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface BarChartProps {
  data: ChartData;
  title: string;
  className?: string;
  horizontal?: boolean;
}

export function BarChart({ data, title, className, horizontal = false }: BarChartProps) {
  // Check if we have multiple datasets with different y-axes
  const hasMultipleYAxes = data.datasets.some(dataset => dataset.yAxisID === 'y1');
  
  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: horizontal ? 'y' : 'x',
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: hasMultipleYAxes ? {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        beginAtZero: true,
        title: {
          display: true,
          text: 'Number of Transactions'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        beginAtZero: true,
        grid: {
          drawOnChartArea: false, // only draw grid lines for the primary y-axis
        },
        title: {
          display: true,
          text: 'Avg. Transaction Amount ($)'
        }
      },
    } : {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <Card title={title} className={className}>
      <div className="h-56">
        <Bar options={options} data={data} />
      </div>
    </Card>
  );
}
