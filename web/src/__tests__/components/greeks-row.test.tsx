import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { GreeksRow } from "@/components/pricing/greeks-row";
import type { PricingResponse } from "@/lib/types";

const mockData: PricingResponse = {
  model: "Black-Scholes",
  price: 6.89,
  delta: 0.5977,
  gamma: 0.027123,
  vega: 0.1894,
  theta: -0.0521,
  rho: 0.2234,
};

describe("GreeksRow", () => {
  it("renders all 5 Greek labels in uppercase", () => {
    render(<GreeksRow data={mockData} />);

    expect(screen.getByText("DELTA")).toBeInTheDocument();
    expect(screen.getByText("GAMMA")).toBeInTheDocument();
    expect(screen.getByText("VEGA")).toBeInTheDocument();
    expect(screen.getByText("THETA")).toBeInTheDocument();
    expect(screen.getByText("RHO")).toBeInTheDocument();
  });

  it("formats delta to 3 decimal places", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("0.598")).toBeInTheDocument();
  });

  it("formats gamma to 3 decimal places", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("0.027")).toBeInTheDocument();
  });

  it("formats vega to 2 decimal places", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("0.19")).toBeInTheDocument();
  });

  it("formats theta with /d suffix", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("-0.052/d")).toBeInTheDocument();
  });

  it("formats rho to 2 decimal places", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("0.22")).toBeInTheDocument();
  });

  it("renders placeholder dashes when data is null", () => {
    render(<GreeksRow data={null} />);

    const dashes = screen.getAllByText("--");
    expect(dashes).toHaveLength(5);
  });

  it("applies custom className", () => {
    const { container } = render(
      <GreeksRow data={mockData} className="my-custom-class" />,
    );
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("my-custom-class");
  });

  it("applies semantic color class for delta", () => {
    render(<GreeksRow data={mockData} />);
    const deltaValue = screen.getByText("0.598");
    expect(deltaValue.className).toContain("text-delta");
  });

  it("applies semantic color class for theta", () => {
    render(<GreeksRow data={mockData} />);
    const thetaValue = screen.getByText("-0.052/d");
    expect(thetaValue.className).toContain("text-theta");
  });

  it("shows zero values with formatGreek precision", () => {
    const zeroData: PricingResponse = {
      ...mockData,
      delta: 0,
    };
    render(<GreeksRow data={zeroData} />);
    const deltaValue = screen.getByText("0.000");
    expect(deltaValue.className).toContain("text-delta");
  });
});
