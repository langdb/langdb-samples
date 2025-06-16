"use client";

import { Metric, ChartData } from '@/lib/metrics';
import { StatCard } from '../ui/StatCard';
import { PieChart } from '../charts/PieChart';

interface KpiSectionProps {
  overallCompletionRate: Metric;
  completionRateByOfferType: ChartData;
}

export function KpiSection({
  overallCompletionRate,
  completionRateByOfferType,
}: KpiSectionProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <StatCard metric={overallCompletionRate} />
      <div className="md:col-span-2">
        <PieChart 
          data={completionRateByOfferType} 
          title="Completion by Offer Type" 
        />
      </div>
    </div>
  );
}
