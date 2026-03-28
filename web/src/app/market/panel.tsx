"use client";

import MarketPage from "./page";

export default function MarketPanel() {
  return (
    <div className="[&>div>div:first-child]:hidden">
      <MarketPage />
    </div>
  );
}
