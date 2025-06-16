"use client";

import { ChartData } from '@/lib/metrics';
import { LineChart } from '../charts/LineChart';
import { BarChart } from '../charts/BarChart';
import { StackedBarChart } from '../charts/StackedBarChart';

interface TrendsSectionProps {
  weeklyAvgTransactions: ChartData;
  weeklyTotalTransactions: ChartData;
  offerTypeDistributionOverTime?: ChartData;
}

export function TrendsSection({
  weeklyAvgTransactions,
  weeklyTotalTransactions,
  offerTypeDistributionOverTime,
}: TrendsSectionProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
      <BarChart 
        data={weeklyAvgTransactions} 
        title="Weekly Avg. Transaction Amount" 
      />
      <BarChart 
        data={weeklyTotalTransactions} 
        title="Weekly Total Transaction Amount" 
      />
      {offerTypeDistributionOverTime && (
        <StackedBarChart 
          data={offerTypeDistributionOverTime} 
          title="Offer Types Over Time" 
        />
      )}
    </div>
  );
}
