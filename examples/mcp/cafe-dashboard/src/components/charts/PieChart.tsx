"use client";

import { ChartData } from '@/lib/metrics';
import { Card } from '../ui/Card';
import { 
  Chart as ChartJS, 
  ArcElement, 
  Tooltip, 
  Legend,
  ChartOptions 
} from 'chart.js';
import { Pie } from 'react-chartjs-2';

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend
);

interface PieChartProps {
  data: ChartData;
  title: string;
  className?: string;
}

export function PieChart({ data, title, className }: PieChartProps) {
  const options: ChartOptions<'pie'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
      },
    },
  };

  return (
    <Card title={title} className={className}>
      <div className="h-56 flex items-center justify-center">
        <Pie options={options} data={data} />
      </div>
    </Card>
  );
}
