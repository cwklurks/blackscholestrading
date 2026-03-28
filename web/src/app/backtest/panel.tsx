"use client";

import BacktestPage from "./page";

export default function BacktestPanel() {
  return (
    <div className="[&>div>div:first-child]:hidden">
      <BacktestPage />
    </div>
  );
}
