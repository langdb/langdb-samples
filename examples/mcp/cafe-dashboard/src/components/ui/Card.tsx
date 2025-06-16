"use client";

export interface CardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function Card({ title, children, className }: CardProps) {
  return (
    <div className={`bg-white rounded-xl shadow-md p-4 overflow-hidden dashboard-card ${className}`}>
      <h3 className="text-lg font-semibold mb-3 text-gray-800">{title}</h3>
      <div className="h-full">{children}</div>
    </div>
  );
}
