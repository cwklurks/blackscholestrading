import { describe, it, expect } from "vitest";
import { formatPrice, formatGreek, formatPnl } from "@/lib/format";

describe("formatPrice", () => {
  it("formats with dollar sign and 2 decimals", () => {
    expect(formatPrice(7.9655)).toBe("$7.97");
  });
  it("handles zero", () => {
    expect(formatPrice(0)).toBe("$0.00");
  });
});

describe("formatGreek", () => {
  it("formats delta with 3 decimals", () => {
    expect(formatGreek("delta", 0.52345)).toBe("0.523");
  });
  it("formats gamma with 3 decimals", () => {
    expect(formatGreek("gamma", 0.01923)).toBe("0.019");
  });
  it("formats theta with /d suffix", () => {
    expect(formatGreek("theta", -0.01274)).toBe("-0.013/d");
  });
  it("formats vega with 2 decimals", () => {
    expect(formatGreek("vega", 18.2345)).toBe("18.23");
  });
  it("formats rho with 2 decimals", () => {
    expect(formatGreek("rho", 12.4567)).toBe("12.46");
  });
});

describe("formatPnl", () => {
  it("formats positive P&L with + sign", () => {
    expect(formatPnl(142.5)).toBe("+$142.50");
  });
  it("formats negative P&L", () => {
    expect(formatPnl(-87.3)).toBe("-$87.30");
  });
  it("formats zero", () => {
    expect(formatPnl(0)).toBe("$0.00");
  });
});
