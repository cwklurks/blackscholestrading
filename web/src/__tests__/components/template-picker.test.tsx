import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TemplatePicker } from "@/components/strategies/template-picker";

describe("TemplatePicker", () => {
  it("renders all 5 strategy templates", () => {
    render(<TemplatePicker onSelect={vi.fn()} />);

    expect(screen.getByText("Straddle")).toBeInTheDocument();
    expect(screen.getByText("Strangle")).toBeInTheDocument();
    expect(screen.getByText("Iron Condor")).toBeInTheDocument();
    expect(screen.getByText("Butterfly")).toBeInTheDocument();
    expect(screen.getByText("Collar")).toBeInTheDocument();
  });

  it("renders template descriptions", () => {
    render(<TemplatePicker onSelect={vi.fn()} />);

    expect(screen.getByText("Long call + long put at ATM")).toBeInTheDocument();
    expect(screen.getByText("Long OTM call + long OTM put")).toBeInTheDocument();
    expect(screen.getByText("Short strangle + long wings")).toBeInTheDocument();
  });

  it("renders the section heading", () => {
    render(<TemplatePicker onSelect={vi.fn()} />);
    expect(screen.getByText("Strategy Templates")).toBeInTheDocument();
  });

  it("calls onSelect with straddle legs (2 legs) when clicked", () => {
    const onSelect = vi.fn();
    render(<TemplatePicker onSelect={onSelect} />);

    fireEvent.click(screen.getByText("Straddle"));

    expect(onSelect).toHaveBeenCalledOnce();
    const legs = onSelect.mock.calls[0][0];
    expect(legs).toHaveLength(2);
    expect(legs).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "call", side: "long", strike: 100 }),
        expect.objectContaining({ type: "put", side: "long", strike: 100 }),
      ]),
    );
  });

  it("calls onSelect with strangle legs (2 legs, OTM strikes)", () => {
    const onSelect = vi.fn();
    render(<TemplatePicker onSelect={onSelect} />);

    fireEvent.click(screen.getByText("Strangle"));

    const legs = onSelect.mock.calls[0][0];
    expect(legs).toHaveLength(2);
    expect(legs).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "call", side: "long", strike: 110 }),
        expect.objectContaining({ type: "put", side: "long", strike: 90 }),
      ]),
    );
  });

  it("calls onSelect with iron condor legs (4 legs)", () => {
    const onSelect = vi.fn();
    render(<TemplatePicker onSelect={onSelect} />);

    fireEvent.click(screen.getByText("Iron Condor"));

    const legs = onSelect.mock.calls[0][0];
    expect(legs).toHaveLength(4);
    // Should have 2 short legs and 2 long legs
    const shortLegs = legs.filter(
      (l: { side: string }) => l.side === "short",
    );
    const longLegs = legs.filter(
      (l: { side: string }) => l.side === "long",
    );
    expect(shortLegs).toHaveLength(2);
    expect(longLegs).toHaveLength(2);
  });

  it("calls onSelect with butterfly legs (3 legs)", () => {
    const onSelect = vi.fn();
    render(<TemplatePicker onSelect={onSelect} />);

    fireEvent.click(screen.getByText("Butterfly"));

    const legs = onSelect.mock.calls[0][0];
    expect(legs).toHaveLength(3);
    // Middle leg should be short with qty 2
    const shortLeg = legs.find(
      (l: { side: string }) => l.side === "short",
    );
    expect(shortLeg).toMatchObject({ strike: 100, qty: 2, side: "short" });
  });

  it("calls onSelect with collar legs (2 legs)", () => {
    const onSelect = vi.fn();
    render(<TemplatePicker onSelect={onSelect} />);

    fireEvent.click(screen.getByText("Collar"));

    const legs = onSelect.mock.calls[0][0];
    expect(legs).toHaveLength(2);
    expect(legs).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "put", side: "long", strike: 95 }),
        expect.objectContaining({ type: "call", side: "short", strike: 105 }),
      ]),
    );
  });

  it("provides copies of legs (not references to internal data)", () => {
    const onSelect = vi.fn();
    render(<TemplatePicker onSelect={onSelect} />);

    fireEvent.click(screen.getByText("Straddle"));
    fireEvent.click(screen.getByText("Straddle"));

    const firstCall = onSelect.mock.calls[0][0];
    const secondCall = onSelect.mock.calls[1][0];

    // Should be equal in value but not the same reference
    expect(firstCall).toEqual(secondCall);
    expect(firstCall[0]).not.toBe(secondCall[0]);
  });

  it("highlights active template when activeName matches", () => {
    render(<TemplatePicker onSelect={vi.fn()} activeName="Straddle" />);

    const straddleButton = screen.getByText("Straddle").closest("button");
    expect(straddleButton?.className).toContain("border-primary");
    expect(straddleButton?.className).toContain("bg-primary");
  });

  it("does not highlight templates when activeName does not match", () => {
    render(<TemplatePicker onSelect={vi.fn()} activeName="Straddle" />);

    const strangleButton = screen.getByText("Strangle").closest("button");
    expect(strangleButton?.className).not.toContain("bg-primary");
  });
});
