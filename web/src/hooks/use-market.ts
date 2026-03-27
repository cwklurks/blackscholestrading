"use client"

import useSWR from "swr"
import { api } from "@/lib/api"
import type { MarketResponse, ChainResponse } from "@/lib/types"

/**
 * SWR hook for GET /api/market/:ticker.
 *
 * Automatically fetches when `ticker` is truthy.
 * Revalidates on focus by default (SWR behaviour).
 *
 * Usage:
 *   const { data, error, isLoading } = useMarket("AAPL")
 */
export function useMarket(ticker: string | null) {
  return useSWR<MarketResponse, Error>(
    ticker ? `/api/market/${ticker}` : null,
    () => api.market(ticker!),
  )
}

/**
 * SWR hook for GET /api/chain/:ticker.
 *
 * Usage:
 *   const { data, error, isLoading } = useChain("AAPL")
 */
export function useChain(ticker: string | null) {
  return useSWR<ChainResponse, Error>(
    ticker ? `/api/chain/${ticker}` : null,
    () => api.chain(ticker!),
  )
}
