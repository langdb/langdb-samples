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

interface StackedBarChartProps {
  data: ChartData;
  title: string;
  className?: string;
}

export function StackedBarChart({ data, title, className }: StackedBarChartProps) {
  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      x: {
        stacked: true,
      },
      y: {
        stacked: true,
        beginAtZero: true
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
