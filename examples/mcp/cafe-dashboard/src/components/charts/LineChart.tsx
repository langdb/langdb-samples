"use client";

import { ChartData } from '@/lib/metrics';
import { Card } from '../ui/Card';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement,
  LineElement, 
  Title, 
  Tooltip, 
  Legend,
  ChartOptions 
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface LineChartProps {
  data: ChartData;
  title: string;
  className?: string;
}

export function LineChart({ data, title, className }: LineChartProps) {
  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <Card title={title} className={className}>
      <div className="h-56">
        <Line options={options} data={data} />
      </div>
    </Card>
  );
}
