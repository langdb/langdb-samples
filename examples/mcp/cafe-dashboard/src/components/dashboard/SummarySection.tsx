"use client";

import { Metric } from '@/lib/metrics';
import { StatCard } from '../ui/StatCard';

interface SummarySectionProps {
  totalTransactions: Metric;
}

export function SummarySection({
  totalTransactions,
}: SummarySectionProps) {
  return (
    <div className="grid grid-cols-1 gap-4 mb-6">
      <StatCard metric={totalTransactions} />
    </div>
  );
}
