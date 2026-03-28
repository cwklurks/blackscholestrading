"use client";

import VolatilityPage from "./page";

export default function VolatilityPanel() {
  return (
    <div className="[&>div>div:first-child]:hidden">
      <VolatilityPage />
    </div>
  );
}
