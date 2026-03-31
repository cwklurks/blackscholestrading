"use client";

import {
  createContext,
  useContext,
  useReducer,
  type Dispatch,
  type ReactNode,
} from "react";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

export interface TickerState {
  ticker: string | null;
  spot: number | null;
  historicalVol: number | null;
}

const INITIAL_STATE: TickerState = {
  ticker: null,
  spot: null,
  historicalVol: null,
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

type TickerAction =
  | { type: "SET_TICKER"; ticker: string }
  | { type: "SET_MARKET_DATA"; ticker: string; spot: number; historicalVol: number }
  | { type: "CLEAR_MARKET" };

function tickerReducer(state: TickerState, action: TickerAction): TickerState {
  switch (action.type) {
    case "SET_TICKER":
      return { ...state, ticker: action.ticker };
    case "SET_MARKET_DATA":
      return {
        ticker: action.ticker,
        spot: action.spot,
        historicalVol: action.historicalVol,
      };
    case "CLEAR_MARKET":
      return INITIAL_STATE;
  }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const TickerStateContext = createContext<TickerState>(INITIAL_STATE);
const TickerDispatchContext = createContext<Dispatch<TickerAction>>(() => {});

export function TickerProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(tickerReducer, INITIAL_STATE);

  return (
    <TickerStateContext.Provider value={state}>
      <TickerDispatchContext.Provider value={dispatch}>
        {children}
      </TickerDispatchContext.Provider>
    </TickerStateContext.Provider>
  );
}

export function useTickerState(): TickerState {
  return useContext(TickerStateContext);
}

export function useTickerDispatch(): Dispatch<TickerAction> {
  return useContext(TickerDispatchContext);
}
