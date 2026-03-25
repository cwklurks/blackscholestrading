"""Multi-leg option strategy for payoff computation."""
import numpy as np

from models.black_scholes import BlackScholesModel


class MultiLegStrategy:
    """Computes P&L for a multi-leg option strategy across a range of spot prices."""

    def __init__(self, legs: list[dict], S: float, T: float = 0.0833,
                 r: float = 0.05, sigma: float = 0.2):
        self.legs = legs
        self.S = S
        self.T = T
        self.r = r
        self.sigma = sigma

    def _leg_entry_price(self, leg: dict) -> float:
        """Get or compute the entry price for a leg."""
        if leg.get("entry_price") is not None:
            return leg["entry_price"]
        # Compute BS price as entry
        bs = BlackScholesModel(self.S, leg["strike"], self.T, self.r, self.sigma)
        if leg["type"] == "call":
            return bs.call_price()
        return bs.put_price()

    def compute_payoff(self, spot_min: float, spot_max: float, steps: int = 200) -> dict:
        """Compute P&L across spot range."""
        prices = np.linspace(spot_min, spot_max, steps)
        total_pnl = np.zeros(steps)

        for leg in self.legs:
            strike = leg["strike"]
            qty = leg.get("qty", 1)
            side_mult = 1 if leg["side"] == "long" else -1
            entry = self._leg_entry_price(leg)

            if leg["type"] == "call":
                intrinsic = np.maximum(prices - strike, 0)
            else:
                intrinsic = np.maximum(strike - prices, 0)

            leg_pnl = side_mult * qty * (intrinsic - entry)
            total_pnl = total_pnl + leg_pnl

        # Find breakevens (where P&L crosses zero)
        breakevens = []
        for i in range(len(total_pnl) - 1):
            if total_pnl[i] * total_pnl[i + 1] < 0:
                # Linear interpolation
                ratio = abs(total_pnl[i]) / (abs(total_pnl[i]) + abs(total_pnl[i + 1]))
                breakevens.append(float(prices[i] + ratio * (prices[i + 1] - prices[i])))

        max_profit = float(np.max(total_pnl))
        max_loss = float(np.min(total_pnl))

        # Check if profit/loss is unbounded (at edges)
        if total_pnl[-1] > total_pnl[-2]:
            max_profit = None  # Unbounded upside
        if total_pnl[0] < total_pnl[1]:
            max_loss = None  # Unbounded downside

        return {
            "prices": prices.tolist(),
            "pnl": total_pnl.tolist(),
            "breakevens": breakevens,
            "max_profit": max_profit,
            "max_loss": max_loss,
        }
