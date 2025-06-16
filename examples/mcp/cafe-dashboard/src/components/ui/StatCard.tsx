"use client";

import { Metric } from '@/lib/metrics';
import { Card } from './Card';

interface StatCardProps {
  metric: Metric;
  className?: string;
}

export function StatCard({ metric, className }: StatCardProps) {
  return (
    <Card title={metric.label} className={className}>
      <div className="flex items-center justify-center h-full">
        <span className="text-3xl font-bold text-indigo-700">{metric.value}</span>
      </div>
    </Card>
  );
}
