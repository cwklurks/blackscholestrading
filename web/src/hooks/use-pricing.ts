"use client"

import useSWRMutation from "swr/mutation"
import { api } from "@/lib/api"
import type { PricingRequest, PricingResponse } from "@/lib/types"

/**
 * SWR mutation hook for POST /api/price.
 *
 * Usage:
 *   const { trigger, data, error, isMutating } = usePricing()
 *   await trigger({ S: 100, K: 100, T: 1, r: 0.05, sigma: 0.2 })
 */
export function usePricing() {
  return useSWRMutation<PricingResponse, Error, string, PricingRequest>(
    "/api/price",
    (_key, { arg }) => api.price(arg),
  )
}
