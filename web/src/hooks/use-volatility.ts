"use client"

import useSWRMutation from "swr/mutation"
import { api } from "@/lib/api"
import type { VolSurfaceRequest, VolSurfaceResponse } from "@/lib/types"

/**
 * SWR mutation hook for POST /api/volatility-surface.
 *
 * Usage:
 *   const { trigger, data, error, isMutating } = useVolSurface()
 *   await trigger({ ticker: "AAPL" })
 */
export function useVolSurface() {
  return useSWRMutation<VolSurfaceResponse, Error, string, VolSurfaceRequest>(
    "/api/volatility-surface",
    (_key, { arg }) => api.volSurface(arg),
  )
}
