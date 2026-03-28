"use client";

import PricingPage from "./page";

export default function PricingPanel() {
  return (
    <div className="[&>div>div:first-child]:hidden">
      <PricingPage />
    </div>
  );
}
