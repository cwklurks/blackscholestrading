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
  it("renders all 5 Greek labels", () => {
    render(<GreeksRow data={mockData} />);

    expect(screen.getByText("Delta")).toBeInTheDocument();
    expect(screen.getByText("Gamma")).toBeInTheDocument();
    expect(screen.getByText("Vega")).toBeInTheDocument();
    expect(screen.getByText("Theta")).toBeInTheDocument();
    expect(screen.getByText("Rho")).toBeInTheDocument();
  });

  it("formats delta to 4 decimal places", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("0.5977")).toBeInTheDocument();
  });

  it("formats gamma to 6 decimal places", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("0.027123")).toBeInTheDocument();
  });

  it("formats vega to 4 decimal places", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("0.1894")).toBeInTheDocument();
  });

  it("displays negative theta with negative sign", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("-0.0521")).toBeInTheDocument();
  });

  it("formats rho to 4 decimal places", () => {
    render(<GreeksRow data={mockData} />);
    expect(screen.getByText("0.2234")).toBeInTheDocument();
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

  it("applies positive color class for positive values", () => {
    render(<GreeksRow data={mockData} />);
    const deltaValue = screen.getByText("0.5977");
    expect(deltaValue.className).toContain("text-positive");
  });

  it("applies negative color class for negative values", () => {
    render(<GreeksRow data={mockData} />);
    const thetaValue = screen.getByText("-0.0521");
    expect(thetaValue.className).toContain("text-negative");
  });

  it("applies muted color class for zero values", () => {
    const zeroData: PricingResponse = {
      ...mockData,
      delta: 0,
    };
    render(<GreeksRow data={zeroData} />);
    const deltaValue = screen.getByText("0.0000");
    expect(deltaValue.className).toContain("text-muted-foreground");
  });
});
