import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock fetch globally before importing the api module
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Import after mocking so the module captures our mock
import { api, ApiError } from "@/lib/api";

beforeEach(() => {
  mockFetch.mockReset();
});

// ---------------------------------------------------------------------------
// POST /api/price
// ---------------------------------------------------------------------------

describe("api.price", () => {
  it("sends correct POST request and returns parsed response", async () => {
    const mockResponse = {
      model: "Black-Scholes",
      price: 6.89,
      delta: 0.59,
      gamma: 0.03,
      vega: 0.19,
      theta: -0.05,
      rho: 0.22,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    const result = await api.price({
      S: 100,
      K: 100,
      T: 0.5,
      r: 0.05,
      sigma: 0.2,
      option_type: "call",
    });

    expect(result).toEqual(mockResponse);
    expect(mockFetch).toHaveBeenCalledOnce();
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/price"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }),
    );

    // Verify the body was serialized correctly
    const callArgs = mockFetch.mock.calls[0];
    const body = JSON.parse(callArgs[1].body);
    expect(body).toMatchObject({ S: 100, K: 100, T: 0.5 });
  });

  it("throws ApiError with detail message on non-ok response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      statusText: "Unprocessable Entity",
      json: () => Promise.resolve({ detail: "Invalid input" }),
    });

    await expect(
      api.price({
        S: -1,
        K: 100,
        T: 0.5,
        r: 0.05,
        sigma: 0.2,
        option_type: "call",
      }),
    ).rejects.toThrow("Invalid input");
  });

  it("throws ApiError with status code", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ detail: "Server error" }),
    });

    try {
      await api.price({ S: 100, K: 100, T: 0.5, r: 0.05, sigma: 0.2 });
      expect.unreachable("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).status).toBe(500);
    }
  });

  it("handles non-JSON error responses gracefully", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: () => Promise.reject(new Error("not JSON")),
    });

    await expect(
      api.price({ S: 100, K: 100, T: 0.5, r: 0.05, sigma: 0.2 }),
    ).rejects.toThrow("API error 502");
  });
});

// ---------------------------------------------------------------------------
// GET /api/market/:ticker
// ---------------------------------------------------------------------------

describe("api.market", () => {
  it("sends GET request with ticker in URL", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          price: 150,
          history: [],
          historical_vol: 0.25,
          fetched_at: "2026-03-26T00:00:00Z",
        }),
    });

    await api.market("AAPL");

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/market/AAPL"),
      undefined,
    );
  });

  it("encodes special characters in ticker", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          price: 50,
          history: [],
          historical_vol: 0.3,
          fetched_at: "",
        }),
    });

    await api.market("BRK.B");

    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain("BRK.B");
  });
});

// ---------------------------------------------------------------------------
// GET /api/chain/:ticker
// ---------------------------------------------------------------------------

describe("api.chain", () => {
  it("sends GET request with ticker in URL", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({ calls: [], puts: [], expirations: [] }),
    });

    await api.chain("TSLA");

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/chain/TSLA"),
      undefined,
    );
  });
});

// ---------------------------------------------------------------------------
// GET /api/health
// ---------------------------------------------------------------------------

describe("api.health", () => {
  it("returns health status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({ status: "ok", models: ["Black-Scholes"] }),
    });

    const result = await api.health();
    expect(result.status).toBe("ok");
    expect(result.models).toContain("Black-Scholes");
  });
});

// ---------------------------------------------------------------------------
// POST /api/heatmap
// ---------------------------------------------------------------------------

describe("api.heatmap", () => {
  it("sends correct heatmap request", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          spot_values: [90, 100, 110],
          vol_values: [0.1, 0.2, 0.3],
          call_prices: [[1, 2, 3]],
          put_prices: [[3, 2, 1]],
        }),
    });

    const result = await api.heatmap({
      K: 100,
      T: 0.5,
      r: 0.05,
      spot_range: { min: 80, max: 120 },
      vol_range: { min: 0.1, max: 0.5 },
    });

    expect(result.spot_values).toHaveLength(3);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/heatmap"),
      expect.objectContaining({ method: "POST" }),
    );
  });
});

// ---------------------------------------------------------------------------
// POST /api/monte-carlo
// ---------------------------------------------------------------------------

describe("api.monteCarlo", () => {
  it("sends correct Monte Carlo request", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          price: 7.12,
          std_error: 0.05,
          terminal_prices: [95, 105, 110],
          confidence_interval: [6.9, 7.3],
        }),
    });

    const result = await api.monteCarlo({
      S: 100,
      K: 100,
      T: 0.5,
      r: 0.05,
      sigma: 0.2,
      paths: 1000,
    });

    expect(result.price).toBe(7.12);
    expect(result.confidence_interval).toHaveLength(2);
  });
});
