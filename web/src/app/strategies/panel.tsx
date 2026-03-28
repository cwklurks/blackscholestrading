"use client";

import StrategiesPage from "./page";

export default function StrategiesPanel() {
  return (
    <div className="[&>div>div:first-child]:hidden">
      <StrategiesPage />
    </div>
  );
}
