"use client";

import { useEffect, useState } from 'react';
import { getAllMetrics } from '@/lib/metrics';
import { KpiSection } from './dashboard/KpiSection';
import { TrendsSection } from './dashboard/TrendsSection';
import { DemographicsSection } from './dashboard/DemographicsSection';
import { SummarySection } from './dashboard/SummarySection';

interface DashboardState {
  loading: boolean;
  error: string | null;
  data: any | null;
}

export function Dashboard() {
  const [state, setState] = useState<DashboardState>({
    loading: true,
    error: null,
    data: null,
  });

  useEffect(() => {
    async function fetchData() {
      try {
        const metrics = await getAllMetrics();
        setState({
          loading: false,
          error: null,
          data: metrics,
        });
      } catch (error) {
        setState({
          loading: false,
          error: 'Failed to load dashboard data. Please try again later.',
          data: null,
        });
        console.error('Error loading dashboard data:', error);
      }
    }

    fetchData();
  }, []);

  if (state.loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="mt-4 text-lg text-gray-700">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  if (state.error) {
    return (
      <div className="bg-red-50 p-6 rounded-lg border border-red-200 text-center">
        <h3 className="text-lg font-semibold text-red-700 mb-2">Error</h3>
        <p className="text-red-600">{state.error}</p>
      </div>
    );
  }

  if (!state.data) {
    return null;
  }

  const { kpis, trends, demographics, summaries } = state.data;

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Key Performance Indicators</h2>
        <KpiSection
          overallCompletionRate={kpis.overallCompletionRate}
          completionRateByOfferType={kpis.completionRateByOfferType}
        />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Performance Trends</h2>
        <TrendsSection
          weeklyAvgTransactions={trends.weeklyAvgTransactions}
          weeklyTotalTransactions={trends.weeklyTotalTransactions}
        />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Customer Demographics</h2>
        <DemographicsSection
          incomeVsCompletionRate={demographics.incomeVsCompletionRate}
        />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Summary Metrics</h2>
        <SummarySection
          totalTransactions={summaries.totalTransactions}
        />
      </section>
    </div>
  );
}
